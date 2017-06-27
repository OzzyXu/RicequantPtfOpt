# 例：类定义及使用
class CAnimal:
    name = 'unname' # 成员变量
    def __init__(self,voice='hello'):    # 重载构造函数
        self.voice = voice            # 创建成员变量并赋初始值
    def __del__(self):              # 重载析构函数
        pass                # 空操作
    def Say(self):
        print(self.voice)

t = CAnimal()        # 定义动物对象t
t.Say()        # t说话

>> hello            # 输出
dog = CAnimal('wow')    # 定义动物对象dog
dog.Say()            # dog说话
>> wow            # 输出




# 例：类的继承
class CAnimal:
    def __init__(self,voice='hello'): # voice初始化默认为hello
        self.voice = voice
    def Say(self):
        print(self.voice)
    def Run(self):
        pass    # 空操作语句（不做任何操作）

class CDog(CAnimal):        # 继承类CAnimal
    def SetVoice(self,voice): # 子类增加函数SetVoice
        self.voice = voice
    def Run(self): # 子类重载函数Run
        print('Running')

bobo = CDog()
bobo.SetVoice('My Name is BoBo!')      # 设置child.data为hello
bobo.Say()
bobo.Run()

>> My Name is BoBo!
>> Running


back1 = equity_list1
len(back1)
# 242


equity_list1 = back1[95:95]+back1[100:202]
len(equity_list1)

# 最终剩余93 可以



equity_list1 = back1[:95] + back1[100:110]



##############
equity_list1  = equity_list6[:104]
portfolio1 = portfolio6
len(equity_list1)

# 90 跑不出来


##

portfolio1 = rqdatac.fund.get_holdings(fund_name, second_period_s).dropna()
equity_list1 = list(portfolio1.order_book_id)
len(equity_list1)


##
equity_list1 = equity_list1[:90]
len(equity_list1)


equity_fund_portfolio_min_variance_risk_parity = pt.TestPortfolio(equity_list1, 'stocks')
equity_fund_portfolio_min_variance_risk_parity.data_clean(equity_list1, first_period_s, first_period_e)
elimination_list = equity_fund_portfolio_min_variance_risk_parity.kickout_list+equity_fund_portfolio_min_variance_risk_parity.st_list + \
                   equity_fund_portfolio_min_variance_risk_parity.suspended_list
inherited_holdings_weights = list(portfolio1.loc[portfolio1['order_book_id'].isin(elimination_list)].weight)
inherited_holdings_weights = [x/100 for x in inherited_holdings_weights]
len(inherited_holdings_weights)

optimal_weights = list(equity_fund_portfolio_min_variance_risk_parity.min_variance_risk_parity_optimizer())
len(optimal_weights)