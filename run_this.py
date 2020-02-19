from Maze import Maze

from RL_brain import DeepQNetwork
from Learning import QNetWork
from  TestCode import QNetWork_test





def run_maze():

    step = 0

    for episode in range(1200):

        # initial observation

        observation = env.reset()



        while True:

            # fresh env

            env.render()



            # RL choose action based on observation

            action = RL.choose_action(observation)



            # RL take action and get next observation and reward

            observation_, reward, done = env.step(action)

            #print(observation,action,reward,observation_)


            RL.store_transition(observation, action, reward, observation_)



            if (step > 200) and (step % 5 == 0):

                RL.learn()



            # swap observation

            observation = observation_



            # break while loop when end of this episode

            if done:

                break

            step += 1



    # end of game

    print('game over')

    env.destroy()





if __name__ == "__main__":

    # maze game

    env = Maze()

    RL = QNetWork(env.n_actions,env.n_features)

    env.after(100, run_maze)

    env.mainloop()

    RL.plot_Loss()