import logging

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def fnRSI(self,m_Df, m_N=15):

    U = np.where(m_Df.diff(1) > 0, m_Df.diff(1), 0)
    D = np.where(m_Df.diff(1) < 0, m_Df.diff(1) *(-1), 0)

    AU = pd.DataFrame(U).rolling( window=m_N, min_periods=m_N).mean()
    AD = pd.DataFrame(D).rolling( window=m_N, min_periods=m_N).mean()
    RSI = AU.div(AD+AU)[0].mean()
    return RSI


class TradingEnv:
    def __init__(self, custom_args, env_id, obs_data_len, step_len, sample_len,
                 df, fee, initial_budget, n_action_intervals, deal_col_name='c',
                 feature_names=['c', 'v'],
                 return_transaction=True, sell_at_end=False,
                 fluc_div=100.0, gameover_limit=5,max_fee_rate=.01,
                 *args, **kwargs):
        """
        # need deal price as essential and specified the df format
        # obs_data_leng -> observation data length
        # step_len -> when call step rolling windows will + step_len
        # sample_len -> length of each sample
        # df -> dataframe that contain data for trading(format as...)
            # price 
            # datetime
            # serial_number -> serial num of deal at each day recalculating

        # fee -> Proportion of price to pay as fee when buying assets;
                 (e.g. 0.5 = 50%, 0.01 = 1%)
        # initial_budget -> The amount of budget to begin with
        # n_action_intervals -> Number of actions for Buy and Sell actions;
                                 The total number of actions is `n_action_intervals` * 2 + 1
        # deal_col_name -> the column name for cucalate reward used.
        # feature_names -> list contain the feature columns to use in trading status.
        # ?day trade option set as default if don't use this need modify
        """
        assert 0 <= fee <= 1, "fee must be between 0 and 1 (0% to 100%)"
        assert deal_col_name in df.columns, "deal_col not in Dataframe please define the correct column name of which column want to calculate the profit."
        for col in feature_names:
            assert col in df.columns, "feature name: {} not in Dataframe.".format(col)

        self.custom_args = custom_args
        self.total_fee = 0

        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        self.logger = logging.getLogger(env_id)
        # self.file_loc_path = os.environ.get('FILEPATH', '')

        self.df = df

        self.n_action_intervals = n_action_intervals
        self.action_space = 2 * n_action_intervals + 1
        self.hold_action = n_action_intervals

        act_desc_pairs = [(n_action_intervals, "Hold")]
        for i in range(n_action_intervals):
            act_desc_pairs.append((i, "Buy with {:.2f}% of current budget".format((i + 1) / n_action_intervals * 100)))
            act_desc_pairs.append((n_action_intervals + i + 1,
                                   "Sell {:.2f}% of current assets".format((i + 1) / n_action_intervals * 100)))
        act_desc_pairs.sort(key=lambda x: x[0])
        self.action_describe = {i: s for i, s in act_desc_pairs}

        self.obs_len = obs_data_len
        self.feature_len = len(feature_names)
        self.observation_space = np.array([self.obs_len * self.feature_len, ])
        self.using_feature = feature_names
        self.price_name = deal_col_name

        self.step_len = step_len
        self.fee_rate = fee

        self.sample_len = sample_len

        self.initial_budget = initial_budget
        self.budget = initial_budget

        self.fluc_div = fluc_div
        self.gameover = gameover_limit  # todo : 이게 뭘까
        self.return_transaction = return_transaction
        self.sell_at_end = sell_at_end

        self.render_on = 0
        self.buy_color, self.sell_color = (1, 2)
        self.new_rotation, self.cover_rotation = (1, 2)
        self.transaction_details = pd.DataFrame()
        self.logger.info('Making new env: {}'.format(env_id))

    def _random_choice_section(self):  # todo : 마치 배치 뽑는거 같은건가?
        begin_point = np.random.randint(len(self.df) - self.sample_len + 1)
        end_point = begin_point + self.sample_len
        df_section = self.df.iloc[begin_point: end_point]
        return df_section

    def reset(self):  # prepares various state components
        self.total_fee = 0
        self.df_sample = self._random_choice_section()

        self.step_st = 0
        # define the price to calculate the reward
        self.price = self.df_sample[self.price_name].as_matrix()
        # define the observation feature
        self.obs_features = self.df_sample[self.using_feature].as_matrix()
        # maybe make market position feature in final feature, set as option
        self.posi_arr = np.zeros_like(self.price)  # 보유 주식수
        # position variation
        self.posi_variation_arr = np.zeros_like(self.posi_arr)  # 보유 주식수의 변화기록
        # position entry or cover :new_entry->1  increase->2 cover->-1 decrease->-2
        self.posi_entry_cover_arr = np.zeros_like(self.posi_arr)  # long 포지션인지 short 포지션인지 기록해둠. 아니면 증감
        # self.position_feature = np.array(self.posi_l[self.step_st:self.step_st+self.obs_len])/(self.max_position*2)+0.5

        self.price_mean_arr = self.price.copy()
        self.reward_fluctuant_arr = (self.price - self.price_mean_arr) * self.posi_arr  # 현재가 - 보유단가 ; 미실현 손익으로 추정함
        self.reward_makereal_arr = self.posi_arr.copy()  # bool 로 추정함. 실제 주식처분했는지 아닌지
        self.reward_arr = self.reward_fluctuant_arr * self.reward_makereal_arr  # (현재가 - 보유단가) * 팔았는지 안팔았는지

        self.budget = self.initial_budget

        self.info = None
        self.transaction_details = pd.DataFrame()

        # observation part
        self.previous_rsi = fnRSI(self.df_sample.iloc[self.step_st: self.step_st + self.obs_len].c)
        self.df_sample['v'].iloc[self.step_st: self.step_st + self.obs_len]
        self.obs_state = self.obs_features[self.step_st: self.step_st + self.obs_len]
        self.obs_posi = self.posi_arr[self.step_st: self.step_st + self.obs_len]
        self.obs_posi_var = self.posi_variation_arr[self.step_st: self.step_st + self.obs_len]
        self.obs_posi_entry_cover = self.posi_entry_cover_arr[self.step_st: self.step_st + self.obs_len]
        self.obs_price = self.price[self.step_st: self.step_st + self.obs_len]
        self.obs_price_mean = self.price_mean_arr[self.step_st: self.step_st + self.obs_len]
        self.obs_reward_fluctuant = self.reward_fluctuant_arr[self.step_st: self.step_st + self.obs_len]
        self.obs_makereal = self.reward_makereal_arr[self.step_st: self.step_st + self.obs_len]
        self.obs_reward = self.reward_arr[self.step_st: self.step_st + self.obs_len]

        if self.return_transaction:
            self.obs_return = np.concatenate((self.obs_state,
                                              self.obs_posi[:, np.newaxis],
                                              self.obs_posi_var[:, np.newaxis],
                                              self.obs_posi_entry_cover[:, np.newaxis],
                                              self.obs_price[:, np.newaxis],
                                              self.obs_price_mean[:, np.newaxis],
                                              self.obs_reward_fluctuant[:, np.newaxis],
                                              self.obs_makereal[:, np.newaxis],
                                              self.obs_reward[:, np.newaxis],
                                              np.array([self.fee_rate for _ in range(self.obs_len)])[:, np.newaxis]),

                                             axis=1)
        else:
            self.obs_return = self.obs_state

        self.t_index = 0
        return self.obs_return

    def _long(self, open_posi, enter_price, current_mkt_position, current_price_mean, action):  # Used once in `step()`
        fee = self.fee_rate * enter_price
        enter_price += fee  # fee = 실제 내는 돈, self.fee_rate = 수수료
        betting_rate = (action + 1) / self.n_action_intervals
        n_stock = self.budget * betting_rate / enter_price  # 주문할 주식 수
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

    def _long_cover(self, current_price_mean, current_mkt_position, action):  # Used once in `step()`
        # n_stock = (보유주식 개수) * (비율(액션))
        n_stock = current_mkt_position * (action - self.hold_action) / self.n_action_intervals
        # n_stock = min(action - self.hold_action, current_mkt_position)
        total_value = self.chg_price[0] * n_stock
        fee = self.fee_rate * total_value
        self.budget += total_value - fee
        self.chg_price_mean[:] = current_price_mean
        self.chg_posi[:] = current_mkt_position - n_stock
        self.chg_makereal[:1] = 1
        self.chg_reward[:] = ((self.chg_price * (
                1 - self.fee_rate) - self.chg_price_mean) * n_stock) * self.chg_makereal / self.initial_budget
        self.chg_posi_var[:1] = -n_stock
        self.chg_posi_entry_cover[:1] = -1

    def _stayon(self, current_price_mean, current_mkt_position):  # Used once in `step()`
        self.chg_posi[:] = current_mkt_position
        self.chg_price_mean[:] = current_price_mean
        
    
    def step(self, action):
        current_index = self.step_st + self.obs_len - 1
        current_price_mean = self.price_mean_arr[current_index]
        current_mkt_position = self.posi_arr[current_index]

        self.t_index += 1
        self.step_st += self.step_len
        # observation part
        self.obs_state = self.obs_features[self.step_st: self.step_st + self.obs_len]
        self.obs_posi = self.posi_arr[self.step_st: self.step_st + self.obs_len]
        # position variation
        self.obs_posi_var = self.posi_variation_arr[self.step_st: self.step_st + self.obs_len]
        # position entry or cover :new_entry->1  increase->2 cover->-1 decrease->-2
        self.obs_posi_entry_cover = self.posi_entry_cover_arr[self.step_st: self.step_st + self.obs_len]
        self.obs_price = self.price[self.step_st: self.step_st + self.obs_len]
        self.obs_price_mean = self.price_mean_arr[self.step_st: self.step_st + self.obs_len]
        self.obs_reward_fluctuant = self.reward_fluctuant_arr[self.step_st: self.step_st + self.obs_len]
        self.obs_makereal = self.reward_makereal_arr[self.step_st: self.step_st + self.obs_len]
        self.obs_reward = self.reward_arr[self.step_st: self.step_st + self.obs_len]
        # change part
        self.chg_posi = self.obs_posi[-self.step_len:]
        self.chg_posi_var = self.obs_posi_var[-self.step_len:]
        self.chg_posi_entry_cover = self.obs_posi_entry_cover[-self.step_len:]
        self.chg_price = self.obs_price[-self.step_len:]
        self.chg_price_mean = self.obs_price_mean[-self.step_len:]
        self.chg_reward_fluctuant = self.obs_reward_fluctuant[-self.step_len:]
        self.chg_makereal = self.obs_makereal[-self.step_len:]
        self.chg_reward = self.obs_reward[-self.step_len:]
        
       
        present_rsi=fnRSI(self.df_sample.iloc[self.step_st: self.step_st + self.obs_len].c)
        self.fee_rate=np.clip(self.fee_rate*present_rsi/self.previous_rsi,0,self.max_fee_rate)

        done = False
        if self.step_st + self.obs_len + self.step_len >= len(self.price):
            done = True
            if current_mkt_position != 0 and self.sell_at_end:
                action = -1
                self.chg_price_mean[:] = current_price_mean
                self.chg_posi[:] = 0
                self.chg_posi_var[:1] = -current_mkt_position
                self.chg_posi_entry_cover[:1] = -2
                self.chg_makereal[:1] = 1
                self.budget += self.chg_price[0] * current_mkt_position
                self.chg_reward[:] = (self.chg_price * (
                        1 - self.fee_rate) - self.chg_price_mean) * current_mkt_position * self.chg_makereal / self.initial_budget
            self.transaction_details = pd.DataFrame([self.posi_arr,
                                                     self.posi_variation_arr,
                                                     self.posi_entry_cover_arr,
                                                     self.price_mean_arr,
                                                     self.reward_fluctuant_arr,
                                                     self.reward_makereal_arr,
                                                     self.reward_arr
                                                    ],
                                                    index=['position', 'position_variation', 'entry_cover',
                                                           'price_mean', 'reward_fluctuant', 'reward_makereal',
                                                           'reward'],
                                                    columns=self.df_sample.index).T
            self.info = self.df_sample.join(self.transaction_details)

        # use next tick, maybe choice avg in first 10 tick will be better to real backtest
        # action 이 0~20으로 들어온다고 가정하겠음 [0,9] -> buy , 10 = hold [11,20] -> sell
        # self.hold_action = 10
        # self.actions = [-1, -0.9, -0.8, ... , -0.1, 0, 0.1, ... , 0.8, 0.9, 1]

        enter_price = self.chg_price[0]
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

        self.chg_reward_fluctuant[:] = (self.chg_price * (
                1 - self.fee_rate) - self.chg_price_mean) * self.chg_posi / self.initial_budget

        if self.return_transaction:
            self.obs_return = np.concatenate((self.obs_state,
                                              self.obs_posi[:, np.newaxis],
                                              self.obs_posi_var[:, np.newaxis],
                                              self.obs_posi_entry_cover[:, np.newaxis],
                                              self.obs_price[:, np.newaxis],
                                              self.obs_price_mean[:, np.newaxis],
                                              self.obs_reward_fluctuant[:, np.newaxis],
                                              self.obs_makereal[:, np.newaxis],
                                              self.obs_reward[:, np.newaxis],
                                              np.array([self.fee_rate for _ in range(self.obs_len)])[:, np.newaxis])
                                             ,
                                             axis=1)
        else:
            self.obs_return = self.obs_state

        return self.obs_return, self.chg_reward[0], done, self.info

    # =====================================================Rendering Stuff=====================================================#

    def _gen_trade_color(self, ind, long_entry=(1, 0, 0, 0.5), long_cover=(1, 1, 1, 0.5)):
        if self.posi_variation_arr[ind] > 0 and self.posi_entry_cover_arr[ind] > 0:
            return long_entry
        else:
            return long_cover

    def _plot_trading(self):
        price_x = list(range(len(self.price[:self.step_st + self.obs_len])))
        self.price_plot = self.ax.plot(price_x, self.price[:self.step_st + self.obs_len], c=(0, 0.68, 0.95, 0.9),
                                       zorder=1)
        # maybe seperate up down color
        # self.price_plot = self.ax.plot(price_x, self.price[:self.step_st+self.obs_len], c=(0, 0.75, 0.95, 0.9),zorder=1)
        # self.features_plot = [self.ax3.plot(price_x, self.obs_features[:self.step_st + self.obs_len, i],
        #                                     c=self.features_color[i])[0] for i in range(self.feature_len)]
        self.posi_plot_long = self.ax3.fill_between(price_x, 0, self.posi_arr[:self.step_st + self.obs_len],
                                                    where=self.posi_arr[:self.step_st + self.obs_len] >= 0,
                                                    facecolor=(1, 0.5, 0, 0.2), edgecolor=(1, 0.5, 0, 0.9), linewidth=1,
                                                    label="posi_plot_long")
        self.ax3.legend(framealpha=0.2, loc="center left")
        rect_high = self.obs_price.max() - self.obs_price.min()
        self.target_box = self.ax.add_patch(
            patches.Rectangle(
                (self.step_st, self.obs_price.min()), self.obs_len, rect_high,
                label='observation', edgecolor=(0.9, 1, 0.2, 0.8), facecolor=(0.95, 1, 0.1, 0.3),
                linestyle='-', linewidth=1.5,
                fill=True)
        )  # remove background)
        self.fluc_reward_plot_p = self.ax2.fill_between(price_x, 0,
                                                        self.reward_fluctuant_arr[:self.step_st + self.obs_len],
                                                        where=self.reward_fluctuant_arr[
                                                              :self.step_st + self.obs_len] >= 0,
                                                        facecolor=(1, 0.8, 0, 0.2), edgecolor=(1, 0.8, 0, 0.9),
                                                        linewidth=0.8, label="fluc_reward_plot_p")
        self.fluc_reward_plot_n = self.ax2.fill_between(price_x, 0,
                                                        self.reward_fluctuant_arr[:self.step_st + self.obs_len],
                                                        where=self.reward_fluctuant_arr[
                                                              :self.step_st + self.obs_len] <= 0,
                                                        facecolor=(0, 1, 0.8, 0.2), edgecolor=(0, 1, 0.8, 0.9),
                                                        linewidth=0.8, label="fluc_reward_plot_n")

        self.reward_plot_p = self.ax2.fill_between(price_x, 0,
                                                   self.reward_arr[:self.step_st + self.obs_len].cumsum(),
                                                   where=self.reward_arr[:self.step_st + self.obs_len].cumsum() >= 0,
                                                   facecolor=(1, 0, 0, 0.2), edgecolor=(1, 0, 0, 0.9), linewidth=1,
                                                   label="reward_plot_p")
        self.reward_plot_n = self.ax2.fill_between(price_x, 0,
                                                   self.reward_arr[:self.step_st + self.obs_len].cumsum(),
                                                   where=self.reward_arr[:self.step_st + self.obs_len].cumsum() <= 0,
                                                   facecolor=(0, 1, 0, 0.2), edgecolor=(0, 1, 0, 0.9), linewidth=1,
                                                   label="reward_plot_n")
        self.ax2.legend(framealpha=0.2, loc="center left")

        trade_x = self.posi_variation_arr.nonzero()[0]
        trade_x_buy = [i for i in trade_x if self.posi_variation_arr[i] > 0]
        trade_x_sell = [i for i in trade_x if self.posi_variation_arr[i] < 0]
        trade_y_buy = [self.price[i] for i in trade_x_buy]
        trade_y_sell = [self.price[i] for i in trade_x_sell]
        trade_color_buy = [self._gen_trade_color(i) for i in trade_x_buy]
        trade_color_sell = [self._gen_trade_color(i) for i in trade_x_sell]
        self.trade_plot_buy = self.ax.scatter(x=trade_x_buy, y=trade_y_buy, s=100, marker='^',
                                              c=trade_color_buy, edgecolors=(1, 0, 0, 0.9), zorder=2)
        self.trade_plot_sell = self.ax.scatter(x=trade_x_sell, y=trade_y_sell, s=100, marker='v',
                                               c=trade_color_sell, edgecolors=(0, 1, 0, 0.9), zorder=2)

    def render(self, save=False):
        if self.render_on == 0:
            matplotlib.style.use('dark_background')
            self.render_on = 1

            left, width = 0.1, 0.8
            rect1 = [left, 0.4, width, 0.55]
            rect2 = [left, 0.2, width, 0.2]
            rect3 = [left, 0.05, width, 0.15]

            self.fig = plt.figure(figsize=(15, 8))
            self.fig.suptitle('%s' % self.df_sample['datetime'].iloc[0].date(), fontsize=14, fontweight='bold')
            # self.ax = self.fig.add_subplot(1,1,1)
            self.ax = self.fig.add_axes(rect1)  # left, bottom, width, height
            self.ax2 = self.fig.add_axes(rect2, sharex=self.ax)
            self.ax3 = self.fig.add_axes(rect3, sharex=self.ax)
            self.ax.grid(color='gray', linestyle='-', linewidth=0.5)
            self.ax2.grid(color='gray', linestyle='-', linewidth=0.5)
            self.ax3.grid(color='gray', linestyle='-', linewidth=0.5)
            # self.features_color = [c.rgb + (0.9,) for c in Color('yellow').range_to(Color('cyan'), self.feature_len)]
            # fig, ax = plt.subplots()
            self._plot_trading()

            self.ax.set_xlim(0, len(self.price[:self.step_st + self.obs_len]) + 200)
            plt.ion()
            # self.fig.tight_layout()
            plt.show()
            if save:
                self.fig.savefig('fig/%s.png' % str(self.t_index))

        elif self.render_on == 1:
            self.ax.lines.remove(self.price_plot[0])
            # [self.ax3.lines.remove(plot) for plot in self.features_plot]
            self.fluc_reward_plot_p.remove()
            self.fluc_reward_plot_n.remove()
            self.target_box.remove()
            self.reward_plot_p.remove()
            self.reward_plot_n.remove()
            self.posi_plot_long.remove()
            self.trade_plot_buy.remove()
            self.trade_plot_sell.remove()

            self._plot_trading()

            self.ax.set_xlim(0, len(self.price[:self.step_st + self.obs_len]) + 200)
            if save:
                self.fig.savefig('fig/%s.png' % str(self.t_index))
            plt.pause(0.0001)
