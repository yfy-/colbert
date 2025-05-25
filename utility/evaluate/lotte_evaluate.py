import argparse
import sys
from collections import defaultdict
import json

from colbert.utils.utils import file_tqdm, print_message


def main(args):
    qid2pids = defaultdict(list)
    with open(args.rankings) as f:
        print_message(f"#> Loading ranked lists from {args.ranking} ..")
        for line in file_tqdm(f):
            if args.trec:
                qid, _, pid, _, _, _ = line.strip().split()
            else:
                qid, pid, _, *score = line.strip().split("\t")

            qid, pid = int(qid), int(pid)
            qid2pids[qid].append(pid)

    num_total_qids = 0
    success = 0
    with open(args.rel_judges) as rjfs:
        for jline in rjfs:
            rjson = json.loads(jline.rstrip())
            qid = rjson["qid"]
            answer_pids = rjson["answer_pids"]
            num_total_qids += 1
            if qid not in qid2pids:
                print(
                    f"WARNING: qid {qid} not found in {args.rankings}!",
                    file=sys.stderr,
                )
                continue
            answer_pids = set(answer_pids)
            if len(set(qid2pids[qid][: args.k]).intersection(answer_pids)) > 0:
                success += 1

    print(f"Success@{args.k}={success / num_total_qids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParse()
    parser.add_argument("rel_judges")
    parser.add_argument("rankings")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--trec", action="store_true")
    args = parser.parse_args()
    exit(main(args))
