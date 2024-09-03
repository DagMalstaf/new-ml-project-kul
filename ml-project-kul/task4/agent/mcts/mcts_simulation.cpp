#include <vector>
#include <random>
#include <iostream>
#include <cmath>

std::mt19937 gen(std::random_device{}());

struct Node {
    std::vector<int> legal_actions;
    std::vector<double> Q;  // Average reward
    std::vector<int> N;     // Visit count
    std::vector<double> W;  // Total reward

    Node(int num_actions) : Q(num_actions, 0.0), N(num_actions, 0), W(num_actions, 0.0) {
        for (int i = 0; i < num_actions; ++i) {
            legal_actions.push_back(i);
        }
    }

    void update_legal_actions(const std::vector<int>& actions) {
        legal_actions.clear();
        for (int action : actions) {
            legal_actions.push_back(action);
        }
    }

    void update_statistics(int action, double reward) {
        if (action < 0 || action >= N.size()) return;
        N[action]++;
        W[action] += reward;
        Q[action] = W[action] / N[action];
    }
};

void run_simulation(Node& node, int num_simulations, double exploration_coefficient) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dist(0, node.legal_actions.size() - 1);

    for (int i = 0; i < num_simulations; ++i) {
        if (node.legal_actions.empty()) continue;
        int action_idx = dist(gen);
        int action = node.legal_actions[action_idx];
        double reward = static_cast<double>(gen() % 2); 
        node.update_statistics(action, reward);
    }
}

extern "C" {
    void simulate(Node* node, int num_simulations, double exploration_coefficient) {
        run_simulation(*node, num_simulations, exploration_coefficient);
    }
}
