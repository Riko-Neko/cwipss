#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

struct Segment {
  std::int64_t start;
  std::int64_t stop;
  double cost;
  double mean;
};

double segment_cost(
    const std::vector<double>& prefix_sum,
    const std::vector<double>& prefix_sq,
    std::int64_t start,
    std::int64_t stop) {
  const auto n = stop - start;
  if (n <= 0) {
    return 0.0;
  }
  const double total = prefix_sum[static_cast<std::size_t>(stop)] - prefix_sum[static_cast<std::size_t>(start)];
  const double total_sq = prefix_sq[static_cast<std::size_t>(stop)] - prefix_sq[static_cast<std::size_t>(start)];
  const double cost = total_sq - total * total / static_cast<double>(n);
  return cost > 0.0 ? cost : 0.0;
}

double segment_mean(const std::vector<double>& prefix_sum, std::int64_t start, std::int64_t stop) {
  const auto n = stop - start;
  if (n <= 0) {
    return 0.0;
  }
  return (prefix_sum[static_cast<std::size_t>(stop)] - prefix_sum[static_cast<std::size_t>(start)]) /
         static_cast<double>(n);
}

std::vector<Segment> build_segments(
    const std::vector<double>& prefix_sum,
    const std::vector<double>& prefix_sq,
    const std::vector<std::int64_t>& previous,
    std::int64_t n) {
  if (previous[static_cast<std::size_t>(n)] < 0) {
    return {Segment{0, n, segment_cost(prefix_sum, prefix_sq, 0, n), segment_mean(prefix_sum, 0, n)}};
  }

  std::vector<std::int64_t> bounds;
  bounds.push_back(n);
  std::int64_t cursor = n;
  while (cursor > 0 && previous[static_cast<std::size_t>(cursor)] >= 0) {
    cursor = previous[static_cast<std::size_t>(cursor)];
    bounds.push_back(cursor);
  }
  std::sort(bounds.begin(), bounds.end());
  bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
  if (bounds.empty() || bounds.front() != 0) {
    bounds.insert(bounds.begin(), 0);
  }

  std::vector<Segment> segments;
  segments.reserve(bounds.size() > 0 ? bounds.size() - 1 : 0);
  for (std::size_t i = 1; i < bounds.size(); ++i) {
    const auto start = bounds[i - 1];
    const auto stop = bounds[i];
    if (stop <= start) {
      continue;
    }
    segments.push_back(
        Segment{start, stop, segment_cost(prefix_sum, prefix_sq, start, stop), segment_mean(prefix_sum, start, stop)});
  }
  return segments;
}

bool contains_candidate(const std::vector<std::int64_t>& candidates, std::int64_t value) {
  return std::find(candidates.begin(), candidates.end(), value) != candidates.end();
}

std::vector<Segment> pelt_exact(
    const std::vector<double>& y,
    double penalty,
    std::int64_t min_size,
    std::int64_t jump) {
  const std::int64_t n = static_cast<std::int64_t>(y.size());
  std::vector<double> prefix_sum(static_cast<std::size_t>(n + 1), 0.0);
  std::vector<double> prefix_sq(static_cast<std::size_t>(n + 1), 0.0);
  for (std::int64_t i = 0; i < n; ++i) {
    const double value = y[static_cast<std::size_t>(i)];
    prefix_sum[static_cast<std::size_t>(i + 1)] = prefix_sum[static_cast<std::size_t>(i)] + value;
    prefix_sq[static_cast<std::size_t>(i + 1)] = prefix_sq[static_cast<std::size_t>(i)] + value * value;
  }

  if (n == 0) {
    return {};
  }
  if (n <= min_size) {
    return {Segment{0, n, segment_cost(prefix_sum, prefix_sq, 0, n), segment_mean(prefix_sum, 0, n)}};
  }

  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> best(static_cast<std::size_t>(n + 1), inf);
  std::vector<std::int64_t> previous(static_cast<std::size_t>(n + 1), -1);
  best[0] = -penalty;
  std::vector<std::int64_t> candidates;
  candidates.push_back(0);

  if (jump <= 1) {
    for (std::int64_t t = min_size; t <= n; ++t) {
      std::vector<std::int64_t> valid;
      valid.reserve(candidates.size());
      for (const auto s : candidates) {
        if (t - s >= min_size) {
          valid.push_back(s);
        }
      }
      if (valid.empty()) {
        candidates.push_back(t - min_size + 1);
        continue;
      }

      double best_cost = inf;
      std::int64_t best_start = valid.front();
      for (const auto s : valid) {
        const double cost = best[static_cast<std::size_t>(s)] + segment_cost(prefix_sum, prefix_sq, s, t) + penalty;
        if (cost < best_cost) {
          best_cost = cost;
          best_start = s;
        }
      }
      best[static_cast<std::size_t>(t)] = best_cost;
      previous[static_cast<std::size_t>(t)] = best_start;

      const double cutoff = best_cost + penalty;
      std::vector<std::int64_t> kept;
      kept.reserve(candidates.size() + 1);
      for (const auto s : candidates) {
        if (t - s < min_size ||
            best[static_cast<std::size_t>(s)] + segment_cost(prefix_sum, prefix_sq, s, t) <= cutoff) {
          kept.push_back(s);
        }
      }
      kept.push_back(t - min_size + 1);
      candidates.swap(kept);
    }
  } else {
    std::vector<std::int64_t> endpoints;
    for (std::int64_t t = jump; t <= n; t += jump) {
      if (t >= min_size) {
        endpoints.push_back(t);
      }
    }
    if (endpoints.empty() || endpoints.back() != n) {
      endpoints.push_back(n);
    }

    for (const auto t : endpoints) {
      std::vector<std::int64_t> valid;
      valid.reserve(candidates.size());
      for (const auto s : candidates) {
        if (t - s >= min_size && std::isfinite(best[static_cast<std::size_t>(s)])) {
          valid.push_back(s);
        }
      }
      if (valid.empty()) {
        if (t < n && !contains_candidate(candidates, t)) {
          candidates.push_back(t);
        }
        continue;
      }

      double best_cost = inf;
      std::int64_t best_start = valid.front();
      for (const auto s : valid) {
        const double cost = best[static_cast<std::size_t>(s)] + segment_cost(prefix_sum, prefix_sq, s, t) + penalty;
        if (cost < best_cost) {
          best_cost = cost;
          best_start = s;
        }
      }
      best[static_cast<std::size_t>(t)] = best_cost;
      previous[static_cast<std::size_t>(t)] = best_start;

      const double cutoff = best_cost + penalty;
      std::vector<std::int64_t> kept;
      kept.reserve(candidates.size() + 1);
      for (const auto s : candidates) {
        if (t - s < min_size ||
            best[static_cast<std::size_t>(s)] + segment_cost(prefix_sum, prefix_sq, s, t) <= cutoff) {
          kept.push_back(s);
        }
      }
      if (t < n && !contains_candidate(kept, t)) {
        kept.push_back(t);
      }
      candidates.swap(kept);
    }
  }

  return build_segments(prefix_sum, prefix_sq, previous, n);
}

py::list pelt_mean_shift(
    py::array_t<double, py::array::c_style | py::array::forcecast> activity,
    double penalty,
    std::int64_t min_size,
    std::int64_t jump) {
  const auto buf = activity.request();
  if (buf.ndim != 1) {
    throw py::value_error("activity must be a 1D array");
  }

  min_size = std::max<std::int64_t>(1, min_size);
  jump = std::max<std::int64_t>(1, jump);
  penalty = std::max(0.0, penalty);

  const auto n = static_cast<std::int64_t>(buf.shape[0]);
  const auto* ptr = static_cast<const double*>(buf.ptr);
  std::vector<double> y;
  y.reserve(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    const double value = ptr[i];
    y.push_back(std::isfinite(value) ? value : 0.0);
  }

  py::list rows;
  for (const auto& segment : pelt_exact(y, penalty, min_size, jump)) {
    rows.append(py::make_tuple(segment.start, segment.stop, segment.cost, segment.mean));
  }
  return rows;
}

py::list pelt_mean_shift_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> activity,
    double penalty,
    std::int64_t min_size,
    std::int64_t jump,
    std::int64_t threads) {
  const auto buf = activity.request();
  if (buf.ndim != 2) {
    throw py::value_error("activity must be a 2D array with shape (channels, records)");
  }

  min_size = std::max<std::int64_t>(1, min_size);
  jump = std::max<std::int64_t>(1, jump);
  penalty = std::max(0.0, penalty);
  const auto channels = static_cast<std::int64_t>(buf.shape[0]);
  const auto records = static_cast<std::int64_t>(buf.shape[1]);
  const auto* ptr = static_cast<const double*>(buf.ptr);

  std::vector<std::vector<Segment>> results(static_cast<std::size_t>(channels));
  const std::int64_t worker_count = std::max<std::int64_t>(1, std::min<std::int64_t>(threads, channels));
  {
    py::gil_scoped_release release;
    std::atomic<std::int64_t> next_channel{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(worker_count));
    for (std::int64_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&, worker]() {
        (void)worker;
        while (true) {
          const std::int64_t channel = next_channel.fetch_add(1);
          if (channel >= channels) {
            break;
          }
          std::vector<double> y;
          y.reserve(static_cast<std::size_t>(records));
          const auto offset = channel * records;
          for (std::int64_t i = 0; i < records; ++i) {
            const double value = ptr[offset + i];
            y.push_back(std::isfinite(value) ? value : 0.0);
          }
          results[static_cast<std::size_t>(channel)] = pelt_exact(y, penalty, min_size, jump);
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
  }

  py::list batch_rows;
  for (const auto& channel_segments : results) {
    py::list rows;
    for (const auto& segment : channel_segments) {
      rows.append(py::make_tuple(segment.start, segment.stop, segment.cost, segment.mean));
    }
    batch_rows.append(rows);
  }
  return batch_rows;
}

}  // namespace

PYBIND11_MODULE(_pelt_ext, m) {
  m.doc() = "Native PELT kernels for cwipss";
  m.def(
      "pelt_mean_shift",
      &pelt_mean_shift,
      py::arg("activity"),
      py::arg("penalty") = 16.0,
      py::arg("min_size") = 384,
      py::arg("jump") = 1);
  m.def(
      "pelt_mean_shift_batch",
      &pelt_mean_shift_batch,
      py::arg("activity"),
      py::arg("penalty") = 16.0,
      py::arg("min_size") = 384,
      py::arg("jump") = 1,
      py::arg("threads") = 1);
}
