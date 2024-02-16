from envs.grid_world import GridWorldGUI
from envs.tic_tac_toe import TicTacToeGUI
from envs.cant_stop import CantStopEnv
from envs.balloon_pop import BalloonPopEnv


def launch(env_name: str):
    if env_name == "grid_world":
        env = GridWorldGUI()
        env.run()

    elif env_name == "tic_tac_toe":
        env = TicTacToeGUI()
        env.run()

    elif env_name == "cant_stop":
        env = CantStopEnv()
        env.play()

    elif env_name == "balloon_pop":
        env = BalloonPopEnv()
        env.play()

    else:
        print("Invalid environment name")
        return


if __name__ == '__main__':
    env_name = input("Enter the name of the environment you want to play (grid_world, tic_tac_toe, cant_stop, "
                     "balloon_pop): ")
    launch(env_name)
