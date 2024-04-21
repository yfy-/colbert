#include <pthread.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <utility>

using score_pid = std::pair<float, int>;

void filter_pids_helper(int ncentroids, int nquery_vectors, int npids,
                        int *pids, float *centroid_scores, int *codes,
                        int64_t *doclens, int64_t *offsets, bool *idx,
                        float *term_ub_scores, float max_att_score,
                        int nfiltered_docs, int *filtered_pids) {
  std::priority_queue<score_pid, std::vector<score_pid>,
                      std::greater<score_pid>>
      heap;
  float theta = -9999;
  for (int i = 0; i < npids; ++i) {
    int pid = pids[i];
    float pid_score = 0.0;
    float pid_max_att_score = max_att_score;
    for (int j = 0; j < nquery_vectors; ++j) {
      if (pid_score + pid_max_att_score <= theta)
        break;

      float emb_score = -9999;
      for (int k = 0; k < doclens[pid]; ++k) {
        int code = codes[offsets[pid] + k];
        emb_score =
            std::max(emb_score, centroid_scores[code * nquery_vectors + j]);
      }

      pid_score += emb_score;
      pid_max_att_score -= term_ub_scores[j];
    }

    if (heap.size() < nfiltered_docs) {
      heap.emplace(pid_score, pid);
    } else if (pid_score > theta) {
      heap.pop();
      heap.emplace(pid_score, pid);
    }

    if (heap.size() == nfiltered_docs)
      theta = heap.top().first;
  }

  int num_ret = heap.size();
  for (int i = num_ret - 1; i > -1; --i) {
    filtered_pids[i] = heap.top().second;
    heap.pop();
  }
}

torch::Tensor
filter_pids(const torch::Tensor pids, const torch::Tensor centroid_scores,
            const torch::Tensor codes, const torch::Tensor doclens,
            const torch::Tensor offsets, const torch::Tensor idx,
            int nfiltered_docs, const torch::Tensor term_ub_scores) {
  auto ncentroids = centroid_scores.size(0);
  auto nquery_vectors = centroid_scores.size(1);
  auto npids = pids.size(0);

  auto pids_a = pids.data_ptr<int>();
  auto centroid_scores_a = centroid_scores.data_ptr<float>();
  auto codes_a = codes.data_ptr<int>();
  auto doclens_a = doclens.data_ptr<int64_t>();
  auto offsets_a = offsets.data_ptr<int64_t>();
  auto idx_a = idx.data_ptr<bool>();
  auto term_ub_scores_a = term_ub_scores.data_ptr<float>();

  int nfinal_filtered_docs =
      std::min(nfiltered_docs / 4, static_cast<int>(npids));
  int final_filtered_pids[nfinal_filtered_docs];
  float max_att_score = term_ub_scores.sum(0).item<float>();
  filter_pids_helper(ncentroids, nquery_vectors, npids, pids_a,
                     centroid_scores_a, codes_a, doclens_a, offsets_a, idx_a,
                     term_ub_scores_a, max_att_score, nfinal_filtered_docs,
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
