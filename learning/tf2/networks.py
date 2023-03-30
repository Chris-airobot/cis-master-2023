from tensorflow import keras
from keras.layers import Dense



# Actor decides what to do based on the current state
class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=512):
        # print(f'current directory is {os.getcwd}')
        super(ActorNetwork,self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation='softmax')

    
    def call(self, state):
        result = self.fc1(state)
        result = self.fc2(result)
        result = self.fc3(result)

        return result


# Critic is used to evaluate the states, as in this state good, means we did good for last move
# if this state is bad, it means we chose bad move last time
class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims = 512, fc2_dims=512):
        super(CriticNetwork,self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(1, activation=None)

    
    def call(self, state):
        result = self.fc1(state)
        result = self.fc2(result)
        result = self.fc3(result)
        
        return result
    
