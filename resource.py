import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import japanize_matplotlib
import matplotlib.colors as mcolors
import seaborn as sns
import openai


def plot_stacked_bar_by_week(df, name, data_label, aggregate_value):
    """
    指定された名前に基づき、週ごとにデータラベルの積み上げ棒グラフを描画する関数。
    """
    # カスタムカラーパレットの作成
    colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys())  # 既存のカラーパレットを結合
    colors = colors[:100]  # 必要な色の数に応じて調整

    # 指定された名前でデータを絞り込む
    df_filtered = df[df['名前'] == name]

    # 集計軸（週）とデータラベル項目でグループ化し、集計数値を合計
    df_grouped = df_filtered.groupby(['週', data_label])[aggregate_value].sum().unstack().fillna(0)

    # 積み上げ棒グラフの描画
    fig, ax = plt.subplots(figsize=(7, 4))  # グラフのサイズを変更
    df_grouped.plot(kind='bar', stacked=True, ax=ax)

    plt.title(f'{name} - {aggregate_value} by Week and {data_label}')
    plt.xlabel('週')
    plt.ylabel(aggregate_value)
    plt.xticks(rotation=45)  # x軸のラベルを45度回転
    plt.legend(title=data_label, loc='upper left', bbox_to_anchor=(1, 1))  # 凡例を右側に配置
    plt.tight_layout()

    # Streamlitでグラフを表示
    st.pyplot(fig)

def plot_stacked_bars_grid(df, name, data_labels, aggregate_values):
    """
    指定された名前に基づき、複数のデータラベルと複数の集計数値に対する積み上げ棒グラフをグリッド形式で描画する関数。
    """
    # カスタムカラーパレットの作成
    colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys())  # 既存のカラーパレットを結合
    colors = colors[:100]  # 必要な色の数に応じて調整

    rows = len(aggregate_values)
    cols = len(data_labels)
    fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), sharex=True, sharey=True)

    legend_fontsize_map = {
        '大項目': 'x-small',
        '中項目': 'x-small',
        '小項目': 'xx-small'
    }

    for i, aggregate_value in enumerate(aggregate_values):
        for j, data_label in enumerate(data_labels):
            ax = axs[i][j] if rows > 1 else axs[j]
            df_filtered = df[df['名前'] == name]
            df_grouped = df_filtered.groupby(['週(str)', data_label])[aggregate_value].sum().unstack(fill_value=0)

            # カスタムカラーパレットを適用
            df_grouped.plot(kind='bar', stacked=True, ax=ax, color=colors[:len(df_grouped.columns)])

            ax.set_title(f'{name} - {aggregate_value} by Week and {data_label}')
            ax.set_xlabel('週(str)')
            ax.set_ylabel(aggregate_value)
            ax.tick_params(axis='x', rotation=45)

            if aggregate_value == '実働工数':
                legend_fontsize = legend_fontsize_map.get(data_label, 'small')
                # 凡例をグラフの外側、上部に配置
                ax.legend(title=data_label, fontsize=legend_fontsize, title_fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
            else:
                ax.legend().set_visible(False)

    plt.tight_layout()

    # Streamlitでグラフを表示
    st.pyplot(fig)

def plot_stacked_bars_by_project(df, name, data_label, aggregate_value):
    """
    指定された名前に基づき、案件ごとの総工数を積み上げグラフで表示
    """
    # カスタムカラーパレットの作成
    colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys())  # 既存のカラーパレットを結合
    colors = colors[:100]  # 必要な色の数に応じて調整

    legend_fontsize_map = {
        '案件': 'x-small',
    }

    df_filtered = df[df['名前'] == name]

    # 案件名でグループ化し、週ごとの総工数を計算
    df_grouped = df_filtered.groupby(['週(str)', data_label])[aggregate_value].sum().unstack(fill_value=0)

    # 積み上げ棒グラフの描画
    fig, ax = plt.subplots(figsize=(10, 4))
    df_grouped.plot(kind='bar', stacked=True, ax=ax, color=colors[:len(df_grouped.columns)])

    ax.set_title(f'{name} - {aggregate_value} by Week and {data_label}')
    ax.set_xlabel('週(str)')
    ax.set_ylabel(aggregate_value)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title=data_label, fontsize=legend_fontsize_map['案件'], loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    # Streamlitでグラフを表示
    st.pyplot(fig)

def analyze_midwork_hours(df, target_name, target_week):
    """
    指定したユーザーと週の中項目別工数を分析し、レーダーチャートで比較する。
    """
    # データセット全体から中項目の完全なリストを取得
    all_mid_items = df['中項目'].unique()
    all_mid_items_sorted = np.sort(all_mid_items)[::-1]  # 降順でソートして逆順にする

    # ステップ1: 対象ユーザーのグレードを取得
    target_grade = df.loc[df['名前'] == target_name, 'グレード'].iloc[0]

    # 対象のグレードと週でデータをフィルタリング
    grouped = df[(df['グレード'] == target_grade) & (df['週'] == target_week)]
    sum_work_hours_by_miditem = grouped.groupby('中項目')['実働工数'].sum()
    count_names_by_miditem = grouped.groupby('中項目')['名前'].nunique()
    same_grade_people = grouped.groupby('名前').nunique()

    # 平均工数を計算し、小数点以下2桁に丸める
    avg_work_hours_by_miditem = (sum_work_hours_by_miditem / len(same_grade_people)).reindex(all_mid_items_sorted, fill_value=0).reset_index()
    avg_work_hours_by_miditem.columns = ['中項目', '平均実働工数']
    avg_work_hours_by_miditem['平均実働工数'] = avg_work_hours_by_miditem['平均実働工数'].round(2)

    # 対象ユーザーの中項目別合計工数を計算し、丸める
    target_user_sum = grouped[grouped['名前'] == target_name].groupby('中項目')['実働工数'].sum().reindex(all_mid_items_sorted, fill_value=0).reset_index()
    target_user_sum.columns = ['中項目', 'ユーザー合計実働工数']
    target_user_sum['ユーザー合計実働工数'] = target_user_sum['ユーザー合計実働工数'].round(2)

    # 差分を計算し、丸める
    comparison_df = pd.merge(target_user_sum, avg_work_hours_by_miditem, on='中項目')
    comparison_df['平均との差分'] = (comparison_df['ユーザー合計実働工数'] - comparison_df['平均実働工数']).round(2)
    comparison_df['差分割合 (%)'] = (comparison_df['ユーザー合計実働工数'] / comparison_df['平均実働工数'] * 100).round(0).fillna(0)

    # 'ユーザー合計実働工数'と'平均実働工数'の両方が0の行を削除
    midtrend_strdata_df = comparison_df[(comparison_df['ユーザー合計実働工数'] != 0) | (comparison_df['平均実働工数'] != 0)]
    midtrend_strdata_df = midtrend_strdata_df[['中項目', '平均との差分', '差分割合 (%)']]

    # コメント付与
    midtrend_strdata = midtrend_strdata_df.to_string(index=False)

    # ChatGPTへの問い合わせ用のプロンプトを作成
    prompt = f"{target_name}の中項目別の工数の平均対比のデータです。差分や割合の大きさに着目して分析してください。　＃データ{midtrend_strdata}"
    messages = [
        {"role": "system", "content": prompt}
    ]

    # ChatGPTへの問い合わせ
    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo",
        #model="ft:gpt-3.5-turbo-0125:personal:0429ab:9JJBZQ4v",
        model="gpt-4-0125-preview",
        messages=messages,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )
    # ChatGPTからの応答を取得
    #comment = response.choices[0].message.content.strip()
    #print(comment)

    # チャートの描画
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    ax2 = fig.add_subplot(1, 2, 2)

    # レーダーチャート
    labels = all_mid_items_sorted
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()  # 各軸の角度

    # 最初の値を繰り返して円形グラフを閉じる
    stats_user = np.concatenate((comparison_df['ユーザー合計実働工数'].values, [comparison_df['ユーザー合計実働工数'].values[0]]))
    stats_avg = np.concatenate((comparison_df['平均実働工数'].values, [comparison_df['平均実働工数'].values[0]]))
    angles += angles[:1]

    ax1.set_ylim(0, 0.5)  # 半径を固定
    ax1.fill(angles, stats_user, color='red', alpha=0.75, label='ユーザー実働工数')
    ax1.fill(angles, stats_avg, color='blue', alpha=0.55, label='平均実働工数')
    ax1.plot(angles, stats_user, color='red', linewidth=0.5)  # ユーザーデータの線プロット
    ax1.plot(angles, stats_avg, color='blue', linewidth=0.5)  # 平均データの線プロット
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, size=10)
    ax1.set_title(f'{target_name}の中項目別実働工数比較 (赤: {target_name}, 青: 同G平均)', size=10)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # 表の出力
    ax2.axis('off')
    table = ax2.table(cellText=comparison_df.values, colLabels=comparison_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1)

    fig.suptitle(f'週：{target_week}', size=12)
    plt.tight_layout()

    # Streamlitでグラフを表示
    st.pyplot(fig)

    return midtrend_strdata

def analyze_bigwork_hours(df, target_name, target_week):
    """
    指定したユーザーと週の大項目別工数を分析し、レーダーチャートで比較する。
    """
    # データセット全体から大項目の完全なリストを取得
    all_mid_items = df['大項目'].unique()
    all_mid_items_sorted = np.sort(all_mid_items)[::-1]  # 降順でソートして逆順にする

    # ステップ1: 対象ユーザーのグレードを取得
    target_grade = df.loc[df['名前'] == target_name, 'グレード'].iloc[0]

    # 対象のグレードと週でデータをフィルタリング
    grouped = df[(df['グレード'] == target_grade) & (df['週'] == target_week)]
    sum_work_hours_by_miditem = grouped.groupby('大項目')['実働工数'].sum()
    count_names_by_miditem = grouped.groupby('大項目')['名前'].nunique()
    same_grade_people = grouped.groupby('名前').nunique()

    # 平均工数を計算し、小数点以下2桁に丸める
    avg_work_hours_by_miditem = (sum_work_hours_by_miditem / len(same_grade_people)).reindex(all_mid_items_sorted, fill_value=0).reset_index()
    avg_work_hours_by_miditem.columns = ['大項目', '平均実働工数']
    avg_work_hours_by_miditem['平均実働工数'] = avg_work_hours_by_miditem['平均実働工数'].round(2)

    # 対象ユーザーの大項目別合計工数を計算し、丸める
    target_user_sum = grouped[grouped['名前'] == target_name].groupby('大項目')['実働工数'].sum().reindex(