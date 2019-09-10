# Introduction to Trading ENV

## INIT

    def __init__(self, custom_args, env_id, obs_data_len, step_len, sample_len,
                     df, fee, initial_budget, n_action_intervals, deal_col_name='c',
                     feature_names=['c', 'v'],
                     return_transaction=True, sell_at_end=False,*args, **kwargs):

obs_data_leng -> observation data length, we used : 192

step_len -> when call step rolling windows will + step_len

sample_len -> length of each sample, we used : 480

df -> pandas dataframe that contain data for trading

fee -> Proportion of price to pay as fee when buying/selling assets;

(e.g. 0.5 = 50%, 0.01 = 1%)

initial_budget -> The amount of budget to begin with

n_action_intervals -> Number of actions for Buy and Sell actions;

The total number of actions is `n_action_intervals` * 2 + 1

deal_col_name -> the column name for cucalate reward used. Default is close price

feature_names -> list contain the feature columns to use in trading status.

return_transaction → return trade history if set True. See README.md

sell_at_end → Sell every asset agent have at the end of the episode if set True

## ACTION

3 actions are possible

1. _long : buy assets with cash
2. _long_cover : sell assets
3. _stay_on : hold

note : short is not allowed in this env

### _long

    def _long(self, open_posi, enter_price, current_mkt_position, current_price_mean, action):  # Used once in `step()`
            fee = self.fee * enter_price
            enter_price += fee
            betting_rate = (action + 1) / self.n_action_intervals
            n_stock = self.budget * betting_rate / enter_price
            self.total_fee += n_stock * fee
            self.budget -= enter_price * n_stock
            if open_posi:
                self.chg_price_mean[:] = enter_price
                self.chg_posi[:] = n_stock
                self.chg_posi_var[:1] = n_stock
                self.chg_posi_entry_cover[:1] = 1
            else:
                after_act_mkt_position = current_mkt_position + n_stock
                self.chg_price_mean[:] = (current_price_mean * current_mkt_position + \
                                          enter_price * n_stock) / after_act_mkt_position
                self.chg_posi[:] = after_act_mkt_position
                self.chg_posi_var[:1] = n_stock
                self.chg_posi_entry_cover[:1] = 2

### _long_cover

    def _long_cover(self, current_price_mean, current_mkt_position, action):  # Used once in `step()`
            # n_stock = (보유주식 개수) * (비율(액션))
            n_stock = current_mkt_position * (action - self.hold_action) / self.n_action_intervals
            # n_stock = min(action - self.hold_action, current_mkt_position)
            total_value = self.chg_price[0] * n_stock
            fee = self.fee * total_value
            self.budget += total_value - fee
            self.total_fee += fee
            self.chg_price_mean[:] = current_price_mean
            self.chg_posi[:] = current_mkt_position - n_stock
            self.chg_makereal[:1] = 1
            self.chg_reward[:] = ((self.chg_price * (1 - self.fee) - self.chg_price_mean) * n_stock) * self.chg_makereal / self.initial_budget
            self.chg_posi_var[:1] = -n_stock
            self.chg_posi_entry_cover[:1] = -1

### _stay_on

    def _stayon(self, current_price_mean, current_mkt_position):  # Used once in `step()`
            self.chg_posi[:] = current_mkt_position
            self.chg_price_mean[:] = current_price_mean

### range of actions


You can adjust the percentage of the asset to sell by n_action_interval argument.


You can adjust the percentage of the asset to sell by n_action_interval argument.


    self.n_action_intervals = n_action_intervals
    self.action_space = 2 * n_action_intervals + 1
    self.hold_action = n_action_intervals

**Example : n_action_interval = 5**

action | Effect

0 | buy asset with 20% of current cash

1 | buy asset with 40% of current cash

2 | buy asset with 60% of current cash

3 | buy asset with 80% of current cash

4 | buy asset with 100% of current cash

5 | hold ; do nothing

6 | sell 20% of current asset 

7 | sell 40% of current asset 

8 | sell 60% of current asset 

9 | sell 80% of current asset 

10  sell 100% of current asset 

## RESET

### state

If the prefix of the word is 'obs', it means that data whose length is obs_len is extracted from the sample data of which the length is sample_len.

If the prefix of the word is 'obs', it means that data whose length is obs_len is extracted from the sample data of which the length is sample_len.

    # observation part
    self.obs_state = self.obs_features[self.step_st: self.step_st + self.obs_len]
    self.obs_posi = self.posi_arr[self.step_st: self.step_st + self.obs_len]
    self.obs_posi_var = self.posi_variation_arr[self.step_st: self.step_st + self.obs_len]
    self.obs_posi_entry_cover = self.posi_entry_cover_arr[self.step_st: self.step_st + self.obs_len]
    self.obs_price = self.price[self.step_st: self.step_st + self.obs_len]
    self.obs_price_mean = self.price_mean_arr[self.step_st: self.step_st + self.obs_len]
    self.obs_reward_fluctuant = self.reward_fluctuant_arr[self.step_st: self.step_st + self.obs_len]
    self.obs_makereal = self.reward_makereal_arr[self.step_st: self.step_st + self.obs_len]
    self.obs_reward = self.reward_arr[self.step_st: self.step_st + self.obs_len]


 The word whose prefix is 'chg' means the record of amount of change.

 The word whose prefix is 'chg' means the record of amount of change.

    # change part
    self.chg_posi = self.obs_posi[-self.step_len:]
    self.chg_posi_var = self.obs_posi_var[-self.step_len:]
    self.chg_posi_entry_cover = self.obs_posi_entry_cover[-self.step_len:]
    self.chg_price = self.obs_price[-self.step_len:]
    self.chg_price_mean = self.obs_price_mean[-self.step_len:]
    self.chg_reward_fluctuant = self.obs_reward_fluctuant[-self.step_len:]
    self.chg_makereal = self.obs_makereal[-self.step_len:]
    self.chg_reward = self.obs_reward[-self.step_len:]

self.obs_features

: Observation features are selected among the features of df.

  

self.posi_arr

: The information of the position at current timestep. ( positive → long, negative → short)
  
 
self.posi_variation_arr

: It indicates which action the agent chosen, and the change of status (Only the first value changes : 1 → long, -1 → short)
 

self.posi_entry_cover 

: It shows the characteristic of the action compared to current position (2 → augment position, 1 → start new position, -1 → clear position)
  

self.price 

: Retrieves the information about current timestep from price column of the dataframe
  
self.price_mean_arr

: The average pirce fo current positions. (The income is calculated by this value)
 

self.reward_fluctuant_arr

: It shows the unrealized gains. When the action is taken, these values change to obs_reward. 

self.reward_makereal_arr

: It shows whether the profit and loss is realized abouth the position at current timestep. ( It gives information only from the first value of step_len)


  It shows whether the profit and loss is realized abouth the position at current timestep. ( It gives information only from the first value of step_len)

Final returned state : concatentation of various features

    #returned state
    self.obs_return = np.concatenate((self.obs_state,
                                                  self.obs_posi[:, np.newaxis],
                                                  self.obs_posi_var[:, np.newaxis],
                                                  self.obs_posi_entry_cover[:, np.newaxis],
                                                  self.obs_price[:, np.newaxis],
                                                  self.obs_price_mean[:, np.newaxis],
                                                  self.obs_reward_fluctuant[:, np.newaxis],
                                                  self.obs_makereal[:, np.newaxis],
                                                  self.obs_reward[:, np.newaxis]), axis=1)

## STEP

    def step(self, action):
      ~~~~~~
      ~~~~~~
    	return self.obs_return, self.chg_reward[0], done, self.info

- Determine action observing current state

    if action < self.hold_action:  # If `Buy`
                if self.budget < enter_price:  # If not enough budget
                    action = self.hold_action
                else:
                    open_posi = (current_mkt_position == 0)
                    self._long(open_posi, enter_price, current_mkt_position, current_price_mean, action)
    
            elif (self.hold_action < action <= self.hold_action + self.n_action_intervals) and (
                    current_mkt_position > 0):  # If `Sell` and `Has Asset`
                self._long_cover(current_price_mean, current_mkt_position, action)
    
            elif (self.hold_action < action <= self.hold_action + self.n_action_intervals) and (
                    current_mkt_position == 0):  # If `Sell` and `No Asset`
                action = self.hold_action
    
            if action == self.hold_action:  # If `Hold`
                if current_mkt_position != 0:
                    self._stayon(current_price_mean, current_mkt_position)

- Return following state and rewards

    #next_state
    self.obs_return = np.concatenate((self.obs_state,
                                                  self.obs_posi[:, np.newaxis],
                                                  self.obs_posi_var[:, np.newaxis],
                                                  self.obs_posi_entry_cover[:, np.newaxis],
                                                  self.obs_price[:, np.newaxis],
                                                  self.obs_price_mean[:, np.newaxis],
                                                  self.obs_reward_fluctuant[:, np.newaxis],
                                                  self.obs_makereal[:, np.newaxis],
                                                  self.obs_reward[:, np.newaxis]), axis=1)
    
    #reward
    self.chg_reward[:] = ((self.chg_price * (1 - self.fee) - self.chg_price_mean) * n_stock) * self.chg_makereal / self.initial_budget
    reward = self.chg_reward[0]

## RENDER

1. ax1 : draws price chart and mark buy / sell by drawing triangles
2. ax2 : draws realized rewards and unrealized rewards
3. ax3: draws volume of the trading and agent's fee