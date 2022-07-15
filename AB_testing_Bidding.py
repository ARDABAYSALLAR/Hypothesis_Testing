# @Author : Arda Baysallar
# AB- testing
#

# CASE ***************************************************************************************
# Facebook recently introduced a new bidding type, 'average bidding',
# as an alternative to the existing bidding type called 'maximum bidding'.
# One of our clients, bombabomba.com, decided to test this new feature and
# would like to do an A/B test to see if average bidding converts more than maximum bidding.
# The A/B test has been going on for 1 month and bombabomba.com is now
# waiting for you to analyze the results of this A/B test.
# The ultimate success criterion for Bombabomba.com is Purchase.
# Therefore, the focus should be on the Purchase metric for statistical testing.


# Explain Data *******************************************************************************
# In this data set, which includes the website information of a company,
# there is information such as the number of advertisements that users see
# and click, as well as earnings information from here. There are two separate
# data sets, the Control and Test group. These datasets are in separate sheets
# of the ab_testing.xlsx excel. Maximum Bidding was applied to the control group
# and Average Bidding was applied to the test group.


# Data Features :
# impressions : Number of ad views
# Click : Number of clicks on the displayed ad
# Purchase : The number of products purchased after the ads clicked
# Earning: Earnings after purchased products


#################################################################################
# Mission 1 : data preparation and analysis
#################################################################################


# ----------------------------------------------------------------------
# IMPORTS :

import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, shapiro, mannwhitneyu, f_oneway, levene, kendalltau, ttest_ind
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.stats.api as sms




# ----------------------------------------------------------------------
xls = pd.ExcelFile('ab_testing.xlsx')
df_c_base = pd.read_excel(xls, 'Control Group')
df_t_base = pd.read_excel(xls, 'Test Group')

# Test Group: Average Bidding
# Control Group: Maximum Bidding
df_c = df_c_base.copy()
df_t = df_t_base.copy()

df_c.head(10)
df_t.head(10)



# ----------------------------------------------------------------------
# Control and Test data analysis
# step 2

# CONTROL
df_c.info()
df_c.size
len(df_c)
df_c.describe().T
# comments : there seems no extravaganza min max problems, outliers
df_c.isnull().sum()
# no missing

# TEST
df_t.info()
df_t.size
len(df_t)
df_t.describe().T
# comments : there seems no extravaganza min max problems, outliers
df_t.isnull().sum()
# no missing
# cross analysis
(df_t.describe().T - df_c.describe().T)
sms.DescrStatsW(df_c['Purchase']).tconfint_mean()
sms.DescrStatsW(df_t['Purchase']).tconfint_mean()


# ----------------------------------------------------------------------
# STEP 3 : concat the frames

pd.set_option("display.max_rows", None)

# CONTROl
Group_Control = np.arange(len(df_c))
Group_Control = pd.DataFrame(Group_Control)
Group_Control[:] = "Control"
Group_Control.rename({0: "Control-Test"}, axis=1, inplace=True)

Control = pd.concat([df_c, Group_Control], axis=1)

# TEST
Group_Test = np.arange(len(df_t))
Group_Test = pd.DataFrame(Group_Test)
Group_Test[:] = "Test"
Group_Test.rename({0: "Control-Test"}, axis=1, inplace=True)

Test = pd.concat([df_t, Group_Test], axis=1)

# GENERAL CONCAT FRAME
Cont_Test = pd.concat([Control, Test])



#################################################################################
# Mission 2 :A-B hypothesis description
#################################################################################
# describing hypothesis
# H0 : M1 == M2
# HA : M1 != M2

# Purchase means test :
Cont_Test.groupby('Control-Test').agg({'Purchase' : ['mean', 'median']})
# It seems like there is a mathematical difference



#################################################################################
# MISSION 3 : AB TEST APPLICATION
#################################################################################


# ----------------------------------------------------------------------
# STEP 1 : ASSUMPTION ANALYSIS


# 1 : NORMAL DISTRIBUTION
# H0 : normal distribution
# H1 : not normal distribution
def norm_test(dataframe, filter_col= 'Control-Test', stat_col = 'Purchase'):
    """
    Normality test results and returns p-values

    :param dataframe: pandas dataframe
    :param filter_col: the column for filtering the two groups
    :param stat_col:  the column to test Shapiro Normality for
    :return:
    Returns dictionary that has keys as values in the column
    and values as p-values per value from shapiro normality test

    example :
    Control-Test column has two unique value Control and Test
    it is binary identifier column
    We used this column as filter - column

    So it is resulted with
    dic['Control'] = p value for Control value from shapiro normality test
    dic['Test'] = p values for Test from value shapiro normality test

    """
    values = list(dataframe[filter_col].unique())
    dic = {}
    for val in values:
        tstat, pval = shapiro(dataframe.loc[dataframe[filter_col] == val, stat_col])
        dic[val] = pval
    return dic


# 2 : VARIANCE HOMOGENIOUS
# H0 : Homogenous variance
# H1 : Non-Homogenous variance
def var_assumption(dataframe,filter_col='Control-Test', stat_col='Purchase') :
    """
    VARIANCE HOMOGENITY TEST WITH LEVENE
        # H0 : Homogenous variance
        # H1 : Non-Homogenous variance
    :param dataframe:  pandas dataframe
    :param filter_col: the column for filtering the two groups
    :param stat_col:  the column to test Levene homogenity for
    :return:
    p-value coming from Levene variance homogenity test
    """
    values = list(dataframe[filter_col].unique())
    tstat, p_val_var_test = levene(dataframe.loc[dataframe[filter_col] == values[0], stat_col],
                            dataframe.loc[dataframe[filter_col] == values[1], stat_col])
    return p_val_var_test


# ALL ASSUMPTION TOGETHER :
# ASSUMPTION CONTROL COMMENT PART
def assumption_result(dataframe, alpha=0.05, filter_col='Control-Test', stat_col='Purchase'):
    """
    According to given dataframe the function will return necessary AB-test comments
    :param
    dataframe: pandas dataframe
    alpha : float, accepted sensitivity parameter, default : 0.05
    :return:
    prints out all the necessary test commentary that could be helpful to identify the reasons fot test decision,
    suggested test and specific parameters that will change according to variance test results
    """
    dictionary = norm_test(dataframe, filter_col, stat_col)
    print("************************ NORMALITY TEST : ************************")
    for k, v in dictionary.items():
        if v < alpha:
            # reject H0: not normal
            print(k, " Column : NOT-NORMAL DISTRIBUTION --> \n" +
                     "Suggested AB test :  Non-parametric(mannwithneyu) test with p-val :", round(v, 3), end = '\n\n')
        print(k, "Column : NORMAL DISTRIBUTION --> \n" +
                 "Suggested AB test : parametric t-test (ttest_ind) with p-val :", round(v, 3), end = '\n\n')
    p_var = var_assumption(dataframe, filter_col, stat_col)
    print("************************ VAR HOMOGENOUS TEST : ************************ ")
    if p_var < alpha:
         # reject H0: not homogenous
        print("Levene Variance homogenity test pval : ", round(p_var, 3), "not homogenous \n")
        print("Suggested AB TEST : ttest_ind T-TEST \n" +
              "with parameter equal_var: False because it is Normal but not homogenous")
    print("Levene Variance homogenity test pval : ", round(p_var, 3), "homogenous \n")
    print("Suggested AB TEST : ttest_ind T-TEST \n" +
          "with parameter equal_var:True because it is Normal and homogenous", end='\n\n')

# ----------------------------------------------------------------------
# STEP 2 : CHOOSING CORRECT TEST ACCORDING TO ASSUMPTION RESULTS  !

# ASSUMPTION CONTROL APPLICATION PART
def assumption_tester(dataframe, alpha=0.05, filter_col='Control-Test', stat_col='Purchase', comment=False):
    """
    Runs assumption tests and returns parametric or non-parametric test flag to decide which to follow
    :param dataframe: pandas dataframe
    :param alpha: sensitivity tolarated
    :param filter_col: binary identifier of groups column
    :param stat_col:  test object column
    :return:
    returns test decision after assumptions are rejected or not with boolean flag values
    if the result is normal and variance is homogeneous -> parametric with equal_var = True
    if the result is normal and variance is not homogeneous -> parametric with equal_var = False
    if the result is not normal non-parametric
    """
    dictionary = norm_test(dataframe, filter_col, stat_col)
    parametric_test_flag = True
    equal_var_flag = True
    if any(val < alpha for val in dictionary.values()):
        # Non - parametric not normal: mannwithneyu
        parametric_test_flag = False
    else:
        if var_assumption(dataframe, filter_col, stat_col) < alpha:
            # H0 reject --> not homogeneous var
            equal_var_flag = False
    if comment:
        assumption_result(dataframe, alpha=alpha, filter_col=filter_col, stat_col=stat_col)

    return parametric_test_flag, equal_var_flag

# AB TESTING APPLICATION PART :
def AB_tester(dataframe, alpha=0.05, filter_col='Control-Test', stat_col='Purchase', comment=False):
    """
    return AB test result(p-value) either if parametric or non-parametric according to assumption results
    :param dataframe: pandas dataframe
    :param alpha: sensisitivity tolareted
    :param filter_col: binary identifier of groups column
    :param stat_col: test object column
    :param comment: Boolean flag deciding commentary returns or not

    :return:
    returns p-value of test result
    """
    parametric_test_flag, equal_var_flag = assumption_tester(dataframe, alpha=alpha,
                                filter_col=filter_col, stat_col=stat_col, comment=comment)
    values = list(dataframe[filter_col].unique())
    if parametric_test_flag:
        ttest_stat, pvalue = ttest_ind(dataframe.loc[dataframe[filter_col] == values[0], stat_col],
                                       dataframe.loc[dataframe[filter_col] == values[1], stat_col],
                                       equal_var=equal_var_flag)
        return pvalue

    ttest_stat, pvalue = mannwhitneyu(dataframe.loc[dataframe[filter_col] == values[0], stat_col],
                                       dataframe.loc[dataframe[filter_col] == values[1], stat_col])
    return pvalue

# AB TESTING APPLICATION WITH RESULT COMMENT PART :

def AB_test_result(pval=AB_tester(Cont_Test) , alpha = 0.05):
    """
    Application function that will give all results of AB testing with commentary
    :param pval: p-value from AB test
    :param alpha: Sensitivity tolarated
    :return:
    Print outs Test results with comment and p-values
    """
    print('************************ RESULT ************************')
    if pval < alpha:
        print('There is significant difference in between group means! pval : ', round(pval, 2))
    else :
        print('There is NOT significant difference in between group means! pval : ', round(pval, 2))





# ----------------------------------------------------------------------
# STEP 3 : APPLICATION AND COMMENT
AB_test_result(pval=AB_tester(Cont_Test,comment=True ) , alpha = 0.05)



#################################################################################
# MISSION 4 : CONCLUSION AND INSIGHTS
#################################################################################
# STEP 1 : WHICH TEST WHY AND WHY ?

o1 = """
We used parametric T-test using ttest_ind from scipy.stats lib
The reason : The distribution was Normal + Variance is Homogenous

"""

o2 = """
Suggestions according to AB Test Results : 
Test : Average Bidding
Control : Maximum Bidding
There is no significant Purchase difference in between Average Bidding and Maximum Bidding Approach
I would suggest to continue with least expensive solution.

When  we check the condidence intervals it seems there is a mathematical difference advantageous 
towards Average Bidding method. However, stastical test shows it might be a luck factor. 

Hence, I suggest to choose least expensive, if they are both equal range , then they can choose 
to go with Average bidding since the condidence interval shows a better interval in between 
530 - 633 rather than Maximum bidding approach's 508 - 593
 
"""

print(o1)
print(o2)