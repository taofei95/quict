#ifndef GRAPH_STRUCTURE_GRAPH_H
#define GRAPH_STRUCTURE_GRAPH_H

#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <sstream>
#include <queue>
#include <set>

namespace graph {
    template<typename vertex_label_t, typename edge_data_t>
    class edge_t {
    public:
        vertex_label_t from_, to_;
        edge_data_t data_;
        size_t next_edge_from_;
        size_t next_edge_to_;

        edge_t(vertex_label_t from, vertex_label_t to, edge_data_t data, size_t next_edge_from, size_t next_edge_to) :
                from_(from), to_(to), data_(data), next_edge_from_(next_edge_from), next_edge_to_(next_edge_to) {}

        edge_t(const edge_t &e) = default;

        void print_info() const {
            std::cout << from_ << " -> " << to_ << ", data = " << data_ << std::endl;
        }

        std::string info_str() const {
            std::stringstream ss;
            ss << from_ << " -> " << to_ << ", data = " << data_ << std::endl;
            return ss.str();
        }
    };


    template<typename vertex_label_t, typename edge_data_t>
    class directed_graph {
    private:
        using my_edge_t = edge_t<vertex_label_t, edge_data_t>;
        using my_shared_edge_t = std::shared_ptr<my_edge_t>;

        std::vector<std::shared_ptr<my_edge_t>> edges_;
        std::map<vertex_label_t, size_t> head_out_;
        std::map<vertex_label_t, size_t> out_deg_;
        std::map<vertex_label_t, size_t> head_in_;
        std::map<vertex_label_t, size_t> in_deg_;
        std::set<vertex_label_t> vertices_;

        class queue_entry_t {
        public:
            vertex_label_t node_;
            edge_data_t dist_;

            queue_entry_t(vertex_label_t node, edge_data_t dist) : node_(node), dist_(dist) {}

            friend bool operator<(const queue_entry_t &a, const queue_entry_t &b) {
                return a.dist_ < b.dist_;
            }

            friend bool operator>(const queue_entry_t &a, const queue_entry_t &b) {
                return a.dist_ > b.dist_;
            }
        };

        template<typename K, typename V>
        inline void set_if_not_present(std::map<K, V> &m, const K &k, const V &v) {
            if (m.find(k) == m.end()) {
                m[k] = v;
            }
        }

    public:
        directed_graph() {
            edges_ = {};
            head_out_ = {};
            head_in_ = {};
            out_deg_ = {};
            in_deg_ = {};
            vertices_ = {};
        }


        void print_info() const {
            std::cout << "----------------------------------------" << std::endl;
            std::cout << "[Vertex]" << std::endl;
            for (const auto&[key, value] : head_out_) {
                std::cout << key << " ";
            }
            std::cout << std::endl << "----------------------------------------" << std::endl;
            std::cout << "[Edge]" << std::endl;
            for (const auto &it:edges_) {
                it->print_info();
            }
            std::cout << "----------------------------------------" << std::endl;
        }

        std::string info_str() const {
            std::stringstream ss;
            ss << "----------------------------------------" << std::endl;
            ss << "[Vertex]" << std::endl;
            for (const auto&[key, value] : head_out_) {
                ss << key << " ";
            }
            ss << std::endl << "----------------------------------------" << std::endl;
            ss << "[Edge]" << std::endl;
            for (const auto &it:edges_) {
//                it->print_info();
                ss << it->info_str();
            }
            ss << "----------------------------------------" << std::endl;
            return ss.str();
        }

        inline void add_vertex(const vertex_label_t &vertex) {
            vertices_.insert(vertex);
            set_if_not_present<vertex_label_t, size_t>(head_out_, vertex, -1);
            set_if_not_present<vertex_label_t, size_t>(head_in_, vertex, -1);
            set_if_not_present<vertex_label_t, size_t>(out_deg_, vertex, 0);
            set_if_not_present<vertex_label_t, size_t>(in_deg_, vertex, 0);
        }

        inline void add_edge(const vertex_label_t &from, const vertex_label_t &to, const edge_data_t &data) {
            add_vertex(from);
            add_vertex(to);

            edges_.emplace_back(std::make_shared<my_edge_t>(from, to, data, head_out_[from], head_in_[to]));

            head_out_[from] = head_in_[to] = edges_.size() - 1;

            out_deg_[from] += 1;
            in_deg_[to] += 1;
        }

        [[nodiscard]] inline size_t edge_cnt() const {
            return edges_.size();
        }

        inline size_t oud_deg_of(const vertex_label_t &nd) const {
            // may throw out_of_range
            return out_deg_.at(nd);
        }

        std::vector<my_shared_edge_t> edges_from(const vertex_label_t &nd) {
            auto ret_vec = std::vector<my_shared_edge_t>();
            // may throw out_of_range
            for (size_t e = head_out_.at(nd); e != -1; e = edges_[e]->next_edge_from_) {
                ret_vec.emplace_back(std::shared_ptr(edges_[e]));
            }
            return ret_vec;  // NRVO here so it would be efficient actually!!!
        }

        inline size_t in_deg_of(const vertex_label_t &nd) const {
            // may throw out_of_range
            return in_deg_.at(nd);
        }

        std::vector<my_shared_edge_t> edges_to(const vertex_label_t &nd) {
            auto ret_vec = std::vector<my_shared_edge_t>();
            // may throw out_of_range
            for (size_t e = head_in_.at(nd); e != -1; e = edges_[e]->next_edge_to_) {
                ret_vec.emplace_back(std::shared_ptr(edges_[e]));
            }
            return ret_vec;
        }

        std::vector<my_shared_edge_t> &edges() {
            return edges_;
        }

        // inf_val should be smaller than maximal value of edge_data_t to
        // avoid overflow.
        std::map<vertex_label_t, edge_data_t>
        dijkstra(vertex_label_t source, edge_data_t inf_val) {
            auto vis = std::map<vertex_label_t, bool>();
            auto dist_map = std::map<vertex_label_t, edge_data_t>();
            for (const auto &it:vertices_) {
                vis[it] = false;
                dist_map[it] = inf_val;
            }

            std::priority_queue<queue_entry_t, std::vector<queue_entry_t>, std::greater<>> q;
            dist_map[source] = 0;
            q.push(queue_entry_t(source, 0));

            while (!q.empty()) {
                auto nd = q.top();
                q.pop();
                auto from = nd.node_;
                if (vis[from]) { continue; }
                for (const auto &e:edges_from(from)) {
                    auto to = e->to_;
                    auto alt = dist_map[from] + e->data_;
                    if (!vis[to] && alt < dist_map[to]) {
                        dist_map[to] = alt;
                        q.push(queue_entry_t(to, alt));
                    }
                }
                vis[from] = true;
            }
            return dist_map;
        }
    };
}

#endif //GRAPH_STRUCTURE_GRAPH_H
