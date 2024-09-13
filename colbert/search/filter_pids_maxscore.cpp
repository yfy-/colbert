#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <utility>

using score_pid = std::pair<float, int>;
using topk_heap = std::priority_queue<score_pid, std::vector<score_pid>,
                                      std::greater<score_pid>>;

typedef struct maxsim_args {
  int tid;
  int nthreads;

  int ncentroids;
  int nquery_vectors;
  int npids;
  int nfiltered_docs;

  int *pids;
  float *centroid_scores;
  int *codes;
  int64_t *doclens;
  int64_t *offsets;
  bool *idx;
  std::vector<float> *cum_max_att_score;
  std::vector<score_pid> local_heap;
} maxsim_args_t;

void sink_heap(std::vector<score_pid> &heap, std::size_t idx) {
  std::size_t heap_size = heap.size();

  while (true) {
    std::size_t min_idx = idx;
    std::size_t l_idx = idx * 2 + 1;
    std::size_t r_idx = l_idx + 1;

    if (l_idx < heap_size && heap[l_idx] < heap[min_idx])
      min_idx = l_idx;

    if (r_idx < heap_size && heap[r_idx] < heap[min_idx])
      min_idx = r_idx;

    if (min_idx == idx)
      break;

    std::swap(heap[idx], heap[min_idx]);
    idx = min_idx;
  }
}

void *maxsim(void *args) {
  maxsim_args_t *maxsim_args = (maxsim_args_t *)args;
  float theta = -9999;
  auto comp = std::greater<score_pid>();
  for (int i = maxsim_args->tid; i < maxsim_args->npids;
       i += maxsim_args->nthreads) {
    int pid = maxsim_args->pids[i];
    float pid_score = 0.0;
    for (int j = 0; j < maxsim_args->nquery_vectors; ++j) {
      if (pid_score + (*maxsim_args->cum_max_att_score)[j] <= theta)
        break;

      float emb_score = -9999;
      for (int k = 0; k < maxsim_args->doclens[pid]; ++k) {
        int code = maxsim_args->codes[maxsim_args->offsets[pid] + k];
        emb_score = std::max(
            emb_score,
            maxsim_args
                ->centroid_scores[code * maxsim_args->nquery_vectors + j]);
      }

      pid_score += emb_score;
    }

    if (maxsim_args->local_heap.size() < maxsim_args->nfiltered_docs) {
      maxsim_args->local_heap.emplace_back(pid_score, pid);
      std::push_heap(std::begin(maxsim_args->local_heap),
                     std::end(maxsim_args->local_heap), comp);
    } else if (pid_score > theta) {
      maxsim_args->local_heap[0] = std::make_pair(pid_score, pid);
      sink_heap(maxsim_args->local_heap, 0);
    }

    if (maxsim_args->local_heap.size() == maxsim_args->nfiltered_docs)
      theta = maxsim_args->local_heap[0].first;
  }

  return nullptr;
}

void filter_pids_helper(int ncentroids, int nquery_vectors, int npids,
                        int *pids, float *centroid_scores, int *codes,
                        int64_t *doclens, int64_t *offsets,
                        std::vector<float> &cum_max_att_score,
                        int nfiltered_docs, int *filtered_pids) {
  auto nthreads = at::get_num_threads();

  pthread_t threads[nthreads];
  maxsim_args_t args[nthreads];
  int thread_nfiltered_docs =
      std::ceil(static_cast<double>(nfiltered_docs) / nthreads);

  for (int i = 0; i < nthreads; i++) {
    args[i].tid = i;
    args[i].nthreads = nthreads;

    args[i].ncentroids = ncentroids;
    args[i].nquery_vectors = nquery_vectors;
    args[i].npids = npids;
    args[i].nfiltered_docs = thread_nfiltered_docs;

    args[i].pids = pids;
    args[i].centroid_scores = centroid_scores;
    args[i].codes = codes;
    args[i].doclens = doclens;
    args[i].offsets = offsets;
    args[i].cum_max_att_score = &cum_max_att_score;

    int rc = pthread_create(&threads[i], NULL, maxsim, (void *)&args[i]);
    if (rc) {
      fprintf(stderr, "Unable to create thread %d: %d\n", i, rc);
      std::exit(1);
    }
  }

  for (int i = 0; i < nthreads; i++) {
    pthread_join(threads[i], NULL);
  }

  auto comp = std::greater<score_pid>();
  float theta = args[0].local_heap[0].first;
  for (int i = 1; i < nthreads; ++i) {
    while (!args[i].local_heap.empty()) {
      const score_pid &top = args[i].local_heap[0];
      if (args[0].local_heap.size() < nfiltered_docs) {
        args[0].local_heap.push_back(args[i].local_heap[0]);
        std::push_heap(std::begin(args[0].local_heap),
                       std::end(args[0].local_heap), comp);
      } else if (top.first > theta) {
        args[0].local_heap[0] = args[i].local_heap[0];
        sink_heap(args[0].local_heap, 0);
      }

      std::pop_heap(std::begin(args[i].local_heap),
                    std::end(args[i].local_heap), comp);
      args[i].local_heap.pop_back();
      if (args[0].local_heap.size() == nfiltered_docs)
        theta = args[0].local_heap[0].first;
    }
  }

  int num_ret = args[0].local_heap.size();
  for (int i = num_ret - 1; i > -1; --i) {
    filtered_pids[i] = args[0].local_heap[0].second;
    std::pop_heap(std::begin(args[0].local_heap), std::end(args[0].local_heap),
                  comp);
    args[0].local_heap.pop_back();
  }
}

torch::Tensor filter_pids(const torch::Tensor pids,
                          const torch::Tensor centroid_scores,
                          const torch::Tensor codes,
                          const torch::Tensor doclens,
                          const torch::Tensor offsets, int nfiltered_docs,
                          const torch::Tensor term_ub_scores) {
  auto ncentroids = centroid_scores.size(0);
  auto nquery_vectors = centroid_scores.size(1);
  auto npids = pids.size(0);

  auto pids_a = pids.data_ptr<int>();
  auto centroid_scores_a = centroid_scores.data_ptr<float>();
  auto codes_a = codes.data_ptr<int>();
  auto doclens_a = doclens.data_ptr<int64_t>();
  auto offsets_a = offsets.data_ptr<int64_t>();
  auto term_ub_scores_a = term_ub_scores.data_ptr<float>();

  std::unordered_set<int> unq_pids;
  std::vector<int> unq_pids_vec;
  for (int i = 0; i < npids; ++i) {
    auto found = unq_pids.find(pids_a[i]);
    if (found == std::end(unq_pids)) {
      unq_pids_vec.push_back(pids_a[i]);
      unq_pids.insert(pids_a[i]);
    }
  }
  npids = unq_pids_vec.size();

  int nfinal_filtered_docs =
      std::min(nfiltered_docs / 4, static_cast<int>(npids));
  int final_filtered_pids[nfinal_filtered_docs];
  float max_att_score = term_ub_scores.sum(0).item<float>();
  std::vector<float> cum_max_att_score(nquery_vectors);
  cum_max_att_score[0] = max_att_score;
  for (int i = 1; i < nquery_vectors; ++i) {
    cum_max_att_score[i] = cum_max_att_score[i - 1] - term_ub_scores_a[i - 1];
  }

  filter_pids_helper(ncentroids, nquery_vectors, npids, unq_pids_vec.data(),
                     centroid_scores_a, codes_a, doclens_a, offsets_a,
                     cum_max_att_score, nfinal_filtered_docs,
                     final_filtered_pids);

  // bool ones[ncentroids];
  // for (int i = 0; i < ncentroids; i++) {
  //   ones[i] = true;
  // }
  // filter_pids_helper(ncentroids, nquery_vectors, nfiltered_docs,
  // filtered_pids,
  //                    centroid_scores_a, codes_a, doclens_a, offsets_a, ones,
  //                    nfinal_filtered_docs, final_filtered_pids);

  auto options =
      torch::TensorOptions().dtype(torch::kInt32).requires_grad(false);
  return torch::from_blob(final_filtered_pids, {nfinal_filtered_docs}, options)
      .clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("filter_pids_cpp", &filter_pids, "Filter pids");
}
