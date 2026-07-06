PYTHON ?= python3
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall

EPISODES ?= 10000
ROWS ?= 9
COLS ?= 9
MINES ?= 10
ACTION_MODE ?= reveal
GAMES ?= 100
RUN_NAME ?=
SAVE_DIR ?=
DEVICE ?=
EXTRA_ARGS ?=
CHECKPOINT ?=
OPTIONAL_ARGS = $(if $(RUN_NAME),--run_name $(RUN_NAME),) $(if $(SAVE_DIR),--save_dir $(SAVE_DIR),) $(if $(DEVICE),--device $(DEVICE),) $(EXTRA_ARGS)

TRAIN_ARGS = --episodes $(EPISODES) --rows $(ROWS) --cols $(COLS) --mines $(MINES) --action_mode $(ACTION_MODE) $(OPTIONAL_ARGS)
PLAY_ARGS = --rows $(ROWS) --cols $(COLS) --mines $(MINES) --action_mode $(ACTION_MODE) --games $(GAMES) $(if $(DEVICE),--device $(DEVICE),)
DQN_CHECKPOINT = $(or $(CHECKPOINT),$(if $(SAVE_DIR),$(SAVE_DIR)/best.pt,checkpoints/dqn/best.pt))
PPO_CHECKPOINT = $(or $(CHECKPOINT),$(if $(SAVE_DIR),$(SAVE_DIR)/best.pt,checkpoints/ppo/best.pt))
GRPO_CHECKPOINT = $(or $(CHECKPOINT),$(if $(SAVE_DIR),$(SAVE_DIR)/best.pt,checkpoints/grpo/best.pt))

.PHONY: train-dqn train-ppo train-grpo train-all resume-dqn resume-ppo resume-grpo resume-all \
        play-dqn play-ppo play-grpo eval-dqn eval-ppo eval-grpo train-bc test clean

minesweeper: minesweeper.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

train-dqn:
	$(PYTHON) train.py --algo dqn $(TRAIN_ARGS)

train-ppo:
	$(PYTHON) train.py --algo ppo $(TRAIN_ARGS)

train-grpo:
	$(PYTHON) train.py --algo grpo $(TRAIN_ARGS)

train-all: train-dqn train-ppo train-grpo

resume-dqn:
	$(PYTHON) train.py --algo dqn $(TRAIN_ARGS) --resume

resume-ppo:
	$(PYTHON) train.py --algo ppo $(TRAIN_ARGS) --resume

resume-grpo:
	$(PYTHON) train.py --algo grpo $(TRAIN_ARGS) --resume

resume-all: resume-dqn resume-ppo resume-grpo

train-bc:
	$(PYTHON) train_bc.py --rows $(ROWS) --cols $(COLS) --mines $(MINES)

play-dqn:
	$(PYTHON) play.py --algo dqn --checkpoint $(DQN_CHECKPOINT) $(PLAY_ARGS)

play-ppo:
	$(PYTHON) play.py --algo ppo --checkpoint $(PPO_CHECKPOINT) $(PLAY_ARGS)

play-grpo:
	$(PYTHON) play.py --algo grpo --checkpoint $(GRPO_CHECKPOINT) $(PLAY_ARGS)

eval-dqn:
	$(PYTHON) play.py --algo dqn --checkpoint $(DQN_CHECKPOINT) $(PLAY_ARGS) --eval

eval-ppo:
	$(PYTHON) play.py --algo ppo --checkpoint $(PPO_CHECKPOINT) $(PLAY_ARGS) --eval

eval-grpo:
	$(PYTHON) play.py --algo grpo --checkpoint $(GRPO_CHECKPOINT) $(PLAY_ARGS) --eval

test:
	$(PYTHON) -m unittest discover -s tests

clean:
	rm -f minesweeper
	rm -rf checkpoints runs .pytest_cache
