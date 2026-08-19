import fnmatch

import torch
import torch.nn as nn

from .data import get_loaders


def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0"), split="test"):
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    dataset = f"wikitext2_{split}"
    print(f"evaluating on WikiText-2 {split} at seqlen={model.seqlen}")
    _, testloader = get_loaders(dataset, nsamples=0, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)
    with torch.no_grad():
        return eval_ppl_wikitext(model, testloader, 1, device)


def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    nsamples = len(trainloader)
    nlls = []
    print(f"nsamples {nsamples}")
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")
        j = min(i + bs, nsamples)
        inputs = trainloader[i][0].to(device).reshape(j - i, model.seqlen)
        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss = nn.CrossEntropyLoss()(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        nlls.append(loss.float() * (model.seqlen - 1) * (j - i))
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (model.seqlen - 1)))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ppl.item()


def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    nlls = []
    print(f"nsamples {nsamples}")
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")
        j = min(i + bs, nsamples)
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)
        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss = nn.CrossEntropyLoss()(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        nlls.append(loss.float() * (model.seqlen - 1) * (j - i))
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (model.seqlen - 1)))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer, task_list=None, num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator
    if task_list is None:
        task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"]

    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = 2000 if "70b" in model_name or "65b" in model_name else None
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    return evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
    )
