"""Offline PyTorch logit distillation with selectable LoRA targets and merged export.

Floating-point students only. No RL, QAT, automatic FP8 export or quality claim.
"""
import argparse
import json
import math
from pathlib import Path
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from common import (fresh_output, load_model, model_logits, read_blocks,
                    validate_splits, write_json)
from matrix import check_linear


class RecoveryLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        check_linear(base)
        if not 0 < rank <= min(base.in_features, base.out_features):
            raise ValueError('Invalid recovery rank')
        self.base = base.requires_grad_(False)
        # Keep optimizer parameters FP32; cast updates only for forward/merge.
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 1.0 / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype)) * self.scale

    def merged(self):
        base = self.base
        with torch.no_grad():
            base.weight.copy_((base.weight.float() + self.scale * (self.b @ self.a)).to(base.weight.dtype))
        return base


def attach(model, targets, rank):
    pattern = re.compile(targets)
    selected = [(n, m) for n, m in model.named_modules() if n and pattern.fullmatch(n)]
    if not selected:
        raise ValueError('No modules match --targets; inspect model.named_modules()')
    for _, m in selected:
        check_linear(m)
        if not 0 < rank <= min(m.in_features, m.out_features):
            raise ValueError('Invalid rank for selected layer')
    model.requires_grad_(False)
    result = {}
    for name, module in selected:
        parent, _, child = name.rpartition('.')
        branch = RecoveryLinear(module, rank)
        setattr(model.get_submodule(parent) if parent else model, child, branch)
        result[name] = branch
    return result


def kd_loss(student, teacher, labels, temperature=1.0, ce_weight=0.0):
    if temperature <= 0 or not 0 <= ce_weight <= 1:
        raise ValueError('Require temperature > 0 and CE weight in [0,1]')
    if student.shape != teacher.shape or student.shape[:-1] != labels.shape:
        raise ValueError('Teacher/student vocabulary and token positions must align')
    s, t = student.float(), teacher.detach().float()
    loss = F.kl_div(F.log_softmax(s/temperature, -1), F.log_softmax(t/temperature, -1),
                    reduction='sum', log_target=True) * temperature**2 / labels.numel()
    if ce_weight:
        loss = (1-ce_weight)*loss + ce_weight*F.cross_entropy(s.reshape(-1, s.shape[-1]), labels.reshape(-1))
    return loss


@torch.no_grad()
def assess(model, rows, device):
    model.eval()
    total, count = 0., 0
    for row in rows:
        logits = model_logits(model, row, device)
        labels = torch.tensor(row[1:], device=device)
        total += F.cross_entropy(logits[0].float(), labels, reduction='sum').item()
        count += len(labels)
    return dict(nll=total/count, ppl=math.exp(total/count), scored_tokens=count)


def branch_state(branches):
    return {f'{name}.{key}': getattr(module, key).detach().cpu().clone()
            for name, module in branches.items() for key in ('a', 'b')}


def restore(branches, values):
    expected = {f'{name}.{key}' for name in branches for key in ('a', 'b')}
    if set(values) != expected:
        raise ValueError('Checkpoint target modules differ')
    with torch.no_grad():
        for name, module in branches.items():
            for key in ('a', 'b'):
                getattr(module, key).copy_(values[f'{name}.{key}'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('teacher', 'student', 'train', 'validation', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--targets', required=True, help='Full regex over module names; MLP and attention/GDN linear layers allowed')
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--steps', type=int, default=32, help='Total steps including resumed steps')
    p.add_argument('--eval-every', type=int, default=8)
    p.add_argument('--max-minutes', type=float, default=20, help='Training-loop budget; load/export time is additional')
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--temperature', type=float, default=1)
    p.add_argument('--ce-weight', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--teacher-device', default=None, help='Defaults to student device; cpu is allowed but slower')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    p.add_argument('--resume', type=Path, help='Trusted checkpoint from this tool; optimizer and data position included')
    a = p.parse_args()
    if a.steps <= 0 or a.eval_every <= 0 or a.max_minutes <= 0 or a.lr <= 0 or a.temperature <= 0 or not 0 <= a.ce_weight <= 1:
        p.error('Invalid step, time, learning-rate or loss settings')
    train, validation = read_blocks(a.train), read_blocks(a.validation)
    validate_splits(train, validation)
    out = fresh_output(a.output, [a.teacher, a.student, a.train, a.validation])
    torch.manual_seed(a.seed)
    teacher_device = a.teacher_device or a.device
    teacher = load_model(a.teacher, teacher_device, a.dtype).requires_grad_(False)
    student = load_model(a.student, a.device, a.dtype)
    branches = attach(student, a.targets, a.rank)
    # Eval mode disables dropout but still allows gradients through LoRA parameters.
    student.eval()
    optimizer = torch.optim.AdamW([v for v in student.parameters() if v.requires_grad], lr=a.lr, weight_decay=0)
    recipe = dict(teacher=str(a.teacher.resolve()), student=str(a.student.resolve()),
                  train=str(a.train.resolve()), validation=str(a.validation.resolve()), targets=a.targets, rank=a.rank, seed=a.seed,
                  lr=a.lr, dtype=a.dtype, temperature=a.temperature, ce_weight=a.ce_weight)
    initial = assess(student, validation, a.device)
    best, best_step, best_nll, completed = branch_state(branches), 0, initial['nll'], 0
    history = []
    if a.resume:
        checkpoint = torch.load(a.resume, map_location='cpu', weights_only=True)
        if checkpoint['recipe'] != recipe:
            raise ValueError('Resume recipe/data differ; original model files must remain unchanged')
        restore(branches, checkpoint['branches'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        completed = checkpoint['step']
        best, best_step, best_nll = checkpoint['best'], checkpoint['best_step'], checkpoint['best_nll']
        torch.set_rng_state(checkpoint['rng'])
        if torch.cuda.is_available() and checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        history = checkpoint['history']
    started = time.monotonic()
    last_loss = None
    for step in range(completed, a.steps):
        row = train[step % len(train)]
        with torch.no_grad():
            target = model_logits(teacher, row, teacher_device).to(a.device)
        prediction = model_logits(student, row, a.device)
        labels = torch.tensor([row[1:]], device=a.device)
        loss = kd_loss(prediction, target, labels, a.temperature, a.ce_weight)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([v for v in student.parameters() if v.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        last_loss = float(loss.detach())
        del loss, target, prediction
        completed = step+1
        timed_out = time.monotonic()-started >= a.max_minutes*60
        if completed % a.eval_every == 0 or completed == a.steps or timed_out:
            quality = assess(student, validation, a.device)
            history.append(dict(step=completed, training_loss=last_loss,
                                elapsed_seconds=time.monotonic()-started, validation=quality))
            if quality['nll'] < best_nll:
                best, best_step, best_nll = branch_state(branches), completed, quality['nll']
            temporary = out/'resume.tmp'
            torch.save(dict(recipe=recipe, branches=branch_state(branches), optimizer=optimizer.state_dict(),
                            step=completed, best=best, best_step=best_step, best_nll=best_nll,
                            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                            history=history), temporary)
            temporary.replace(out/'resume.pt')
            write_json(out/'progress.json', dict(completed=completed, best_step=best_step, history=history))
        if timed_out:
            break
    restore(branches, best)
    branch_quality = assess(student, validation, a.device)
    for name, module in branches.items():
        parent, _, child = name.rpartition('.')
        setattr(student.get_submodule(parent) if parent else student, child, module.merged())
    merged_quality = assess(student, validation, a.device)
    student.save_pretrained(out/'model', safe_serialization=True)
    write_json(out/'result.json', dict(recipe=recipe, completed=completed, best_step=best_step,
               initial_validation=initial, branch_validation=branch_quality, merged_validation=merged_quality,
               trainable_parameters=sum(v.numel() for m in branches.values() for v in (m.a, m.b)),
               selected_modules=list(branches), history=history,
               status='Floating-point merged export. Fresh reload, independent PPL and actual decode required; requantization is a separate experiment.'))


if __name__ == '__main__':
    main()
