# Env
* make an environment
```python
import gym
env = gym.make('CartPole-v0')
```
* `env.reset()`: reset the new environment (at the beginning of epsiode)
* `env.render()`: display the current state automatically
  * **Note**: when the env return done, can't render anymore, must reset
* `obs, rew, done, info = env.step(action)`: step the action for the env
return observation/reward/done.
* `env.action_space`: 
  * `env.action_space.n`: return the number of choices for action
  * `env.action_space.sample()`: randomly sample an action from n choices

# Space
* the first class in gym, usage `from gym import spaces`
* instantiate a space
```python
space = spaces.Discrete(8) # Set with 8 elements [0, 1, 2 ... 7]
x = space.sample()
assert space.contains(x)
assert space.n == 8
```
* space type: `Discrete`, `Box`: array

# Record & Upload
```python
import gym
env = gym.make('some-env')
env.monitor.start('path/to/monitor')
for i in range(episode_number):
    observation = env.reset()
    action = YourModel(observation)
    obs, rew, done, info = env.step(action)
    if done:
        print ("finished")
        break
env.monitor.close()
```
then you can upload the results to OpenAI Gym:
```
gym.upload('path/to/monitor', api_key='YOUR_API_KEY')
```

