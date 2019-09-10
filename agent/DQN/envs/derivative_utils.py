
import numpy as np
import pandas as pd
def get_stochastic(df, n=15, m=5, t=3):
    # highest price during n days
    ndays_high = df.h.rolling(window=n, min_periods=1).max()
    # lowest price during n days
    ndays_low = df.l.rolling(window=n, min_periods=1).min()

    # Fast%K 
    kdj_k = ((df.c - ndays_low) / (ndays_high - ndays_low))
    # Fast%D (=Slow%K) 
    kdj_d = kdj_k.ewm(span=m).mean()
    # Slow%D 
    kdj_j = kdj_d.ewm(span=t).mean()

    return kdj_j.mean()

def fnRSI(m_Df, m_N=15):
    m_Df = m_Df.c
    U = np.where(m_Df.diff(1) > 0, m_Df.diff(1), 0)
    D = np.where(m_Df.diff(1) < 0, m_Df.diff(1) *(-1), 0)

    AU = pd.DataFrame(U).rolling( window=m_N, min_periods=m_N).mean()
    AD = pd.DataFrame(D).rolling( window=m_N, min_periods=m_N).mean()
    RSI = AU.div(AD+AU)[0].mean()
    return RSI


def get_bollinger_diffs(df, n=20, k=2):
    ma_n = df['c'].rolling(n).mean()
    Bol_upper = df['c'].rolling(n).mean() + k* df['c'].rolling(n).std()
    Bol_lower = df['c'].rolling(n).mean() - k* df['c'].rolling(n).std()
    return (Bol_upper - Bol_lower).mean()

def fnMACD(m_Df, m_NumFast=12, m_NumSlow=26, m_NumSignal=9):
    EMAFast = m_Df['c'].ewm( span = m_NumFast, min_periods = m_NumFast - 1).mean()
    EMASlow = m_Df['c'].ewm( span = m_NumSlow, min_periods = m_NumSlow - 1).mean()
    MACD = EMAFast - EMASlow
    MACDSignal= MACD.ewm( span = m_NumSignal, min_periods = m_NumSignal-1).mean()
    MACDDiff= MACD - MACDSignal
    return MACDDiff.mean()