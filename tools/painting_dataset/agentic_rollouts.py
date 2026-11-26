"""Batched turn-barrier rollouts with private episodes and serialized GPU image/render work."""
import argparse
import contextlib
import json
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from tools.painting_dataset.agentic_episode import EpisodeConfig, run_episode
from tools.painting_dataset.agentic_providers import FrozenEncoderClient, SharedInferenceGenerator
from tools.painting_dataset.agentic_run import CONTRACT
from tools.painting_dataset.painting_reward import score
from tools.painting_dataset.rgb_renderer import factory


def collect_rollouts(*, jobs, output, encoder, generator, batch_size, config, renderer_factory=factory, reward_fn=score):
    """Only main thread calls inference; worker threads own disjoint episode trees."""
    pending = queue.Queue()
    encode_lock, render_lock = threading.Lock(), threading.Lock()
    timing = {"encoding_seconds": 0., "rendering_seconds": 0., "generation_batches": []}

    def encode(messages, directory):
        with encode_lock:
            start = time.monotonic()
            result = encoder(messages, directory)
            timing["encoding_seconds"] += time.monotonic() - start
            return result

    def generate(encoded, request):
        future = Future()
        pending.put((encoded, request, future))
        return future.result()

    @contextlib.contextmanager
    def render_factory(workspace):
        with renderer_factory(workspace) as renderer:
            class LockedRenderer:
                def render(self, *args, **kwargs):
                    with render_lock:
                        start = time.monotonic()
                        result = renderer.render(*args, **kwargs)
                        timing["rendering_seconds"] += time.monotonic() - start
                        return result
            yield LockedRenderer()

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        workers = [pool.submit(run_episode, root=output, target=Path(job["target"]), contract=CONTRACT + "\n" + job.get("instruction", ""),
                               encoder=encode, generator=generate, renderer_factory=render_factory,
                               reward_fn=reward_fn, seed=job["seed"], policy=generator.policy, config=config) for job in jobs]
        try:
            for turn in range(config.turns):
                wave = []
                while len(wave) < len(jobs):
                    try:
                        wave.append(pending.get(timeout=.5))
                    except queue.Empty:
                        for worker in workers:
                            if worker.done() and worker.exception() is not None:
                                raise worker.exception()
                # Stable seed order makes grouping independent of thread/encoder scheduling.
                wave.sort(key=lambda item: item[1]["seed"])
                for offset in range(0, len(wave), batch_size):
                    group = wave[offset:offset+batch_size]
                    begin = time.monotonic()
                    responses = generator.generate_batch([(e, r) for e, r, _ in group])
                    timing["generation_batches"].append({"turn": turn, "size": len(group),
                        "seconds": time.monotonic()-begin, "tokens": sum(len(r["completion_ids"]) for r in responses)})
                    for (_, _, future), response in zip(group, responses, strict=True):
                        future.set_result(response)
                (output / "timing.json").write_text(json.dumps(timing, indent=2))
            episodes = [str(worker.result()) for worker in workers]
        except BaseException as error:
            # Release all pending workers, including workers still encoding, before pool join.
            for _, _, future in locals().get("wave", []):
                if not future.done():
                    future.set_exception(error)
            while not all(w.done() for w in workers):
                try:
                    _, _, future = pending.get(timeout=.5)
                    future.set_exception(error)
                except queue.Empty:
                    pass
            raise
    timing["wall_seconds"] = time.monotonic() - start
    timing["episodes"] = episodes
    (output / "timing.json").write_text(json.dumps(timing, indent=2))
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path, help="JSON list of explicit target paths, unique seeds and optional instructions")
    parser.add_argument("output", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--encoder-gpu", default="1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--history-policy", choices=["full-v1", "bounded-v2"], default="full-v1")
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    if not jobs or len({j["seed"] for j in jobs}) != len(jobs) or args.batch_size < 1:
        raise ValueError("Need nonempty jobs with unique seeds and positive batch size")
    args.output.mkdir(exist_ok=False)
    (args.output / "jobs.json").write_text(json.dumps(jobs, indent=2))
    with FrozenEncoderClient(args.output, args.encoder_gpu) as encoder:
        generator = SharedInferenceGenerator(max_model_len=12288, kv_cache_gib=36, tensor_parallel_size=2,
                                             max_num_seqs=args.batch_size, adapter=args.adapter,
                                             policy="compiled" if args.adapter else "base")
        result = collect_rollouts(jobs=jobs, output=args.output, encoder=encoder, generator=generator,
                                  batch_size=args.batch_size, config=EpisodeConfig(turns=args.turns, max_tokens=8192, history_policy=args.history_policy))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
