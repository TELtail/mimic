import matplotlib.pyplot as plt
import os
from logging import getLogger,config
import numpy as np
from sklearn.metrics import mean_squared_error
from common_utils import log_start


def plot_regression_inference_result(t_test,t_pred,out_path):
    #最終テストの結果の散布図を作成
    result_fig = plt.figure(figsize=(12,9))
    result_ax = result_fig.add_subplot(111)
    result_ax.plot(t_test, t_test, color = 'red') # 直線y = x (真値と予測値が同じ場合は直線状に点がプロットされる)
    result_ax.scatter(t_test, t_pred) # 散布図のプロット
    plt.rcParams["font.size"] = 30
    result_ax.set_xlabel('Correct Answer Label') # x軸ラベル
    result_ax.set_ylabel('Predicted Label') # y軸ラベル
    plt.savefig(os.path.join(out_path,"predict_result.png")) 

def plot_classification_correct_result(correct_train,correct_test,out_path):
    result_fig = plt.figure(figsize=(12,9))
    result_ax = result_fig.add_subplot(111)
    result_ax.plot(correct_train,label="Train Accuracy")
    result_ax.plot(correct_test,label="Test Accuracy")
    plt.rcParams["font.size"] = 30
    result_ax.set_xlabel("Epoch")
    result_ax.set_ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_path,"accuracy_result.png")) 


def plot_loss_glaph(epoch_loss,out_path):
    #損失の変遷をプロット
    labels = ["train","test"]
    epoch_loss = np.array(epoch_loss) #スライスできるようにndarrayに変更
    fig_loss = plt.figure(figsize=(12,8))
    ax_loss = fig_loss.add_subplot(111)
    for i in range(2): #学習データとテストデータのlossだから2
        ax_loss.plot(range(len(epoch_loss)),epoch_loss[:,i],label=labels[i])
    png_path = os.path.join(out_path,"loss.png")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    plt.rcParams["font.size"] = 20
    plt.tight_layout()
    plt.legend()
    fig_loss.savefig(png_path)

def plot_age_histogram(data_t,out_path):
    #年齢の分布をプロット
    labels = np.array(data_t)
    labels = np.ravel(labels) #一次元化
    hist_png_path = os.path.join(out_path,"age_hist.png")
    fig_age = plt.figure(figsize=(12,8))
    ax_age = fig_age.add_subplot(111)
    ax_age.hist(labels,bins=70)
    ax_age.set_xlabel("Age")
    ax_age.set_ylabel("Number of people")
    plt.rcParams["font.size"] = 30
    fig_age.savefig(hist_png_path)
    #mse_to_comparison(labels)

def mse_to_comparison(labels):
    logger = log_start()
    #一様分布、正規分布でのMSEの比較
    uniform = np.random.randint(20,90,len(labels))
    normal = np.random.normal(70,20,len(labels))
    all = np.full(len(labels),70)
    mse_uniform = mean_squared_error(labels,uniform)
    mse_normal = mean_squared_error(labels,normal)
    mse_all = mean_squared_error(labels,all)
    logger.info("All 70: {}".format(mse_all))
    logger.info("Uniform distribution   age U(20,90): {}".format(mse_uniform))
    logger.info("Normal distribution    age N(70,20^2): {}".format(mse_normal))
    logger.info("---------------------------------")

