
import click
import pickle
import os
import numpy as np
import time
import torch.multiprocessing as multiprocessing
from const import *
from torch.multiprocessing import Queue
from models.helper import save_checkpoint, init_models
from lib.agent_play import PybulletGame
from models.controller import Controller

def compute_ranks(x):
  """
  [0, len(x)] におけるランクを返す。
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks


def rankmin(x):
  """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y


def init_controller(controller, solution):
    """  コントローラーの重みをCMAが提案するものに変更する """

    new_w1 = torch.tensor(solution, dtype=torch.double, device=DEVICE)
    params = controller.state_dict() 
    params['fc1.weight'].data.copy_(new_w1.view(-1, PARAMS_CONTROLLER))
    return


def create_results(result_queue, fitlist):
    """  結果キューを空にしてfitlstを適応させ、表示用のタイマーを取得する。"""

    times = []
    for i in range(POPULATION):
        result = result_queue.get()
        keys = list(result.keys())
        result = list(result.values())
        fitlist[keys[0]] = result[0][0]
        times.append(result[0][1])
    return times


def train_controller(current_time):
    """
    CMA-ESアルゴリズムを用いてコントローラをトレーニングし、マルチプロセッシングを用いた並列テストにより候補解を改善する
    """

    current_time = str(current_time)
    number_generations = 1
    games = GAMES
    levels = LEVELS
    current_game = False
    result_queue = Queue()

    vae, lstm, best_controller, solver, checkpoint = init_models(current_time, sequence=1,
                                        load_vae=True, load_controller=True, load_lstm=True)
    if checkpoint:
        current_ctrl_version = checkpoint["version"]
        current_solver_version = checkpoint["solver_version"]
        new_results = solver.result()
        current_best = new_results[1]
    else:
        current_ctrl_version = 1
        current_solver_version = 1
        current_best = 0

    while True:
        solutions = solver.ask()
        fitlist = np.zeros(POPULATION)
        eval_left = 0
        ## Once a level is beaten, remove it from the training set of levels
        if current_best > SCORE_CAP or not current_game:
            if not current_game or len(levels[current_game]) == 0:
                current_game = games[0]
                games.remove(current_game)
                current_best = 0
            current_level = np.random.choice(levels[current_game])
            levels[current_game].remove(current_level)

        print("[CONTROLLER] Current game: %s and level is: %s" % (current_game, current_level))
        while eval_left < POPULATION:
            jobs = []
            todo = PARALLEL if eval_left + PARALLEL <= POPULATION else (eval_left + POPULATION) - PARALLEL
            for i in range(eval_left, todo):
                init_controller(best_controller, solutions[i])
                p = multiprocessing.Process(target=PybulletGame, args=(best_controller, current_level, result_queue))
                jobs.append(p)
                p.start()
                eval_left += 1
            for proc in jobs:
                proc.join()
            times = create_results(result_queue, fitlist)
            solver.tell(solutions, fitlist)
            print("[CONTROLLER] Generation %d, best fitness %f, sigma %f, time %f sec" % (number_generations, solver.result()[1], solver.result()[2], sum(times)))
            number_generations += 1
            save_checkpoint(current_time, current_ctrl_version, current_solver_version, best_controller, vae, lstm, solver, sequence=1)
