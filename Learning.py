import numpy as np
import tensorflow as tf
class QNetWork:
    def __init__(self,n_actions,n_features):
        self.gamma = 0.9
        self.n_features = n_features
        self.n_actions = n_actions
        self.learningRate = 0.01
        self.memory_size = 500
        self.circleNumberstep = 300
        self.LearningstepNumber = 0
        self.epislon = 0.9  #e-greedy
        self.batch_size =32
        self.cost = []
        self.sess = tf.Session()
        self._build_net()
        t_para = tf.get_collection('parameter_target')
        e_para = tf.get_collection('parameters_eval')
        self.replace = [tf.assign(t,e) for t,e in zip(t_para,e_para)]
        self.memory = np.zeros([self.memory_size,self.n_features*2+2])
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        # --------------------eval net-----------------#
        self.initstate = tf.placeholder(tf.float32,[None,self.n_features],name='initstate')   #Input element
        self.q_target = tf.placeholder(tf.float32,[None,self.n_actions],name = 'q_target')   #prepare for calculating the loss value
        with tf.variable_scope('eval_net'):
            collection_name,cell_number,w_initializer,b_initializer = ['parameters_eval',tf.GraphKeys.GLOBAL_VARIABLES],10,tf.random_normal_initializer(0.,0.3),tf.constant_initializer(0.1)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1',[self.n_features,cell_number],initializer=w_initializer,collections=collection_name)
                b1 = tf.get_variable('b1',[1,cell_number],initializer=b_initializer,collections=collection_name)
                #The output of layer 1
                l1 = tf.nn.relu(tf.matmul(self.initstate,w1)+b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2',[cell_number,self.n_actions],initializer=w_initializer,collections=collection_name)
                b2 = tf.get_variable('b2',[1,self.n_actions],initializer=b_initializer,collections=collection_name)
                #the output of layer 2
                self.q_eval = tf.matmul(l1,w2)+b2

            #Calculate the loss
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))

            #Train the net model
            with tf.variable_scope('train'):
                self.train_op = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        #----------------------target net------------------------#
        self.state_prime = tf.placeholder(tf.float32,[None,self.n_features],name="state_prime")

        with tf.variable_scope('target_net'):
            collection_name = ['parameter_target', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, cell_number], initializer=w_initializer,collections=collection_name)
                b1 = tf.get_variable('b1', [1, cell_number], initializer=b_initializer, collections=collection_name)
                # The output of layer 1
                l1 = tf.nn.relu(tf.matmul(self.state_prime, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2',[cell_number,self.n_actions],initializer=w_initializer,collections=collection_name)
                b2 = tf.get_variable('b2',[1,self.n_actions],initializer=b_initializer,collections=collection_name)
                #the output of layer 2
                self.q_next = tf.matmul(l1,w2)+b2

    def store_transition(self,s,a,r,s_prime):
        if not hasattr(self,'memoryCounter'):
            self.memoryCounter = 0
        transition = np.hstack((s,[a,r],s_prime))
        #print(transition)
        #print(self.memoryCounter)

        index = self.memoryCounter % self.memory_size
        self.memory[index,:] = transition
        self.memoryCounter += 1

    def choose_action(self,observation):

        observation = observation[np.newaxis,:]
        if np.random.uniform() < self.epislon:
            action_value = self.sess.run(self.q_eval,feed_dict={self.initstate:observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0,self.n_actions)
        return action

    def learn(self):
        if self.LearningstepNumber % self.circleNumberstep == 0:
            self.sess.run(self.replace)
            print("\n\n\n\nReplace has finished! this is No."+str(self.LearningstepNumber // self.circleNumberstep)+"times\n\n")
        if self.memoryCounter > self.memory_size:
            learnSample = np.random.choice(self.memory_size,size=self.batch_size)
        else:
            learnSample = np.random.choice(self.memoryCounter,size=self.batch_size)

        batch_memory = self.memory[learnSample,:]

        q_next,q_eval = self.sess.run([self.q_next,self.q_eval],feed_dict={self.state_prime:batch_memory[:,-self.n_features:],self.initstate:batch_memory[:,:self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        eval_action_index = batch_memory[:,self.n_features].astype(int)
        reward = batch_memory[:,self.n_features+1]

        q_target[batch_index,eval_action_index] = reward + self.gamma * np.max(q_next,axis=1)
        self.LearningstepNumber += 1

        _,self.tempLoss = self.sess.run([self.train_op,self.loss],feed_dict={self.initstate:batch_memory[:,:self.n_features],self.q_target:q_target})

        self.cost.append(self.tempLoss)

    def plot_Loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost)),self.cost)
        plt.xlabel('training step')
        plt.ylabel('loss')
        plt.show()