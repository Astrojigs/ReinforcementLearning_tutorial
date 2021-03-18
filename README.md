# Code Description
The code in the **main.py file**  contains description on the topic of reinforcement learning given in the Oreilly Book.

The example is: Cartpole

The observations(obs) are *position*, *velocity*, *angle of pole*, *angular velocity* resp.

`env` and `obs` are from **gym**

One can list all the environments possible through OpenAI gym, using;

`gym.envs.registry.all()`

```
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1
totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards+=reward
        if done:
            break
    totals.append(episode_rewards)
```

`print(np.mean(totals),np.std(totals),np.min(totals),np.max(totals))`
`Answer: 41.818 8.908247639126339 24.0 68.0`

Even with **500** tries, this policy never managed to keep the pole upright for more than 68 consecutive steps.

 Let's see if a neural network can come up with a new and better policy

 Create a neural network policy; observations as input and it will output actions to be executed.

 It will estimate the probability of each action.

 In case of CartPole environment there are only two possible actions (left or right), so we only need
 one output neuron. I will output the probability *'p'* of action 0 (left), and of course there
 probability of actions 1 (right) will be *'1-p'*.

 For example: if it outputs **0.7**, then we will pick action 0 with 70% probability, or
 action 1 with 30% probability.

 # Making the neural network:
```
 # Code for neural network policy:
 n_inputs = 4 # == env.observation_space.shape[0]
 # Creating model
 model = tf.keras.models.Sequential()
 model.add(tf.keras.layers.Dense(5, activation='elu', input_shape=[n_inputs]))
 model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
 ```

 <ins>**Note**</ins>: If there were more than two possible actions, there would be one output neuron per action, and we would use the softmax activation function instead.

 If we knew what the best action was at each step, we could train the neural network as usual, by minimizing the cross entropy between the estimated probability distribution and the target distribution. It would be a regular supervised learning.

 However, in Reinforcement Learning the only guidance the agent gets is through rewards, and the rewards are typically sparse and delayed.
 ( for more information, see page 619)

 Tackling the problem of delay, a common strategy is to evaluate an action based on the sum of all the rewards that come after it, usually applying a **discount factor** y (gamma) at each step.

 This sum of discounted rewards is called the *action's return*.

 Consider the example; if an agent decides to go right three times in a row and gets +10 reward after the first step, 0 after the second step, and finally -50 after the third step, then assuming we use a discount factor of `y=0.8`, the first actions will have a return of `10 + y x 0 + y^2 x (-50) = -22`. If the discount factor is close to 0, then future rewards won't count for much compared to immediate rewards. Conversely, if the discount factor is close to 1, then rewards far into the future will count almost as much as the immediate reward. *Typical discount factors vary from 0.9 to 0.99*. With a discount factor of 0.95, rewards 13 steps into the future count roughly for half as much as immediate rewards (since 0.95^13 ~ 0.5), while with a discount factor of 0.99, rewards 69 steps into the future count for half as much as immediate rewards. In the CartPole environment, actions have fairly short-term effects, so choosing a discount factor of 0.95 seems reasonable.

A good actions may be followed by several bad actions that cause the pole to fall quickly, resulting in the good action getting a low return. However, if we play the game enough times, on average good actions will get a higher return than bad ones.

We want to estimate how good or bad an actions is compared to other actions. This is called **action advantage**. For this we must run many episodes and normalize all the action returns(*by subtracting the mean and dividing by the standard deviation*). After that, we can reasonably assume that actions with a negative advantage were bad while actions with a positive advantage were good.

**We are now ready to train our first agent using policy gradients**.

# Policy Gradients:
PG algorithms optimize the parameters of a policy by following the gradients towards higher rewards. One of the popular class of PG algorithms, called *REINFORCE* algorithms, was introduced back in 1992 by Ronald Williams. Here is one common variant.

<ol>
<li>First, let the neural network policy play the game several times, and at each step, compute the gradients that would make the chosen action even more likely-- but don't apply these gradients yet.</li>
<li>Once you have run several episodes, compute each action's advantage(using the method described in the previous section).</li>
<li>If an action's advantage is positive, it means that the action was probably good, and you want to apply the gradients computed earlier to make the action even more likely to be chosen in the future. However, if the action's advantage is negative, it means the action was probably bad, and you want to apply opposite gradients to make this action slightly less likely in the future. The solution is simply to multiply each gradient vector by the corresponding action's advantage.</li>
<li>Finally, compute the mean of all the resulting gradient vectors, and use it to perform a Gradient Descent step.</li>
</ol>

Let's use `tf.keras` to implement this algorithm. We will train the neural network policy we built earlier so that it learns to balance the pole on the cart. First, we need a function that will play one step. We will pretend for now that whatever action it takes is the right one so that we can compute the loss and its gradients (these gradients will just be saved for while, and we will modify them later depending on how good or bad the action turned out to be):

```
def play_one step(env,obs,model,loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis]) #obs[np.newaxis] and obs.reshape((1,)+ obs.shape[:]) is the same
        action = (tf.random.uniform([1,1])>left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action,tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target,left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0,0].numpy()))
    return obs, reward, done, grads
```

Let's walk through this function:

<ul>
<li>Within the *GradientTape* block, we start by calling the model, giving it a single observation (we reshape the observation so it becomes a batch containing a single instance, as the model expects a batch). This outputs the probability of going left.</li>
<li>Next, we sample a random float between 0 ad 1, and we check whether it is greater than `left_proba`. The action will be <ins>False</ins> with probability **`left_proba`** or <ins>True</ins> with **probability 1 - `left_proba`**. Once we cast this Boolean to a number, the action will be 0 (left) or 1 (right) with the appropriate probabilities.</li>
<li>Next, we define the target probability of going left: it is 1 minus the action (cast to a float). If the action is 0 (left), then the target probability of going left will be 1. If the action is 1 (right), then the target probability will be 0</li>
<li>Then we compute the loss using the given loss function, and we use the tape to compute the gradient of the loss with regard to the model's trainable variables. Again, these gradients will be tweaked later, before we apply them, depending on how good or bad the action turned out to be.</li>
<li>Finally, we play the selected action, and we return the new observation, the reward, whether the episode is ended or not, and of course the gradients that we just computed.</li>
</ul>

## Function for playing multiple episodes:
This function will return all the rewards and gradients for each episode and each step:

```
def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads, = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads
```
This code returns a list of reward lists (one reward list per episode, containing one reward per step), as well as a list of gradients lists (one gradient list per episode, each containing one tuple of gradients per step and each tuple containing one gradient tensor per trainable variable).

The algorithm will use the `play_multiple_episodes()` function to play the game several times (e.g., 10 times), then it will go back and look at all the rewards, discount them, and normalize them. To do thatm we need a couple more functions: the first will compute the sum of future discounted rewards at each step, and the second will normalize all these discounted rewards (returns) across many episodes by subtracting the mean and dividing by the standard deviation:

```
# Function to discount
def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step]+=discounted[step + 1] * discount_factor
    return discounted

# Function to normalize rewards:
def discount_and_normalize_rewards(all_rewards, discounted_factor):
    all_discounted_rewards = [discount_rewards(rewards, discounted_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    rewards_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - rewards_mean)/reward_std for discount_rewards in all_discounted_rewards]

```
Calling the `discount_rewards([10, 0, -50], discount_factor=0.8)` returns exactly what we expected in the earlier paragraph.
