import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objects as go
import numpy as np
from openai import OpenAI

import os


# APIキーとスプレッドシートID、シート名を設定
API_KEY = 'AIzaSyBP9qP9XZh1Nm2jsi_MvcWKmTaVNM6F-7A'
SPREADSHEET_ID = '1gvWMXKEEL8OiZlevD3FwmW1b5-lQFlMR_uCv4NvrlDo'
SHEET1_NAME = '過去（実績）'


def load_data():
    # シート1のデータを読み込む
    url_sheet1 = f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{SHEET1_NAME}?key={API_KEY}"
    response_sheet1 = requests.get(url_sheet1)
    data_sheet1 = response_sheet1.json()

    # シート1のデータをPandas DataFrameに変換
    # シート1はカラム名が12行目にあるため、最初の11行をスキップ
    df_sheet1 = pd.DataFrame(data_sheet1['values'][12:])
    df_sheet1.columns = data_sheet1['values'][11]
    df_sheet1 = df_sheet1.reset_index(drop=True)

    # 実働工数の列を数値型に変換し、数値に変換できない値はNaNにする
    # 実働工数が0より大きい行のみを残す（0.00またはNaNを削除）
    df_sheet1['実働工数'] = pd.to_numeric(df_sheet1['実働工数'], errors='coerce')
    df_sheet1 = df_sheet1[df_sheet1['実働工数'] > 0]

    # 名前と週でグループ化し、実働工数と成長工数を合計
    df_grouped = df_sheet1.groupby(['名前', '業務', '案件', 'MGR', 'グレード', '週', '大項目', '中項目', '小項目'])[['実働工数']].sum().reset_index()

    # コピーを作成
    df_filtered = df_grouped.copy()

    # 日付の列を datetime 形式に変換（サンプルでは月日のみのため、年を2024年と仮定）
    df_filtered['週'] = pd.to_datetime(df_filtered['週'].apply(lambda x: f"2024{x}"), format='%Y%m月%d日', errors='coerce')

    # 新しく '週(str)' 列を追加し、日付を文字列形式で格納
    df_filtered['週(str)'] = df_filtered['週'].dt.date

    return df_filtered

def plot_stacked_bar_chart(df, name, category):
    # 指定された名前でデータをフィルタリング
    filtered_df = df[df['名前'] == name]

    # 週の形式を変更
    filtered_df['週'] = pd.to_datetime(filtered_df['週'], format='%Y-%m-%d')
    filtered_df['週'] = filtered_df['週'].dt.strftime('%Y-%m-%d')

    # 週とカテゴリでグループ化し、実働工数を合計
    grouped_df = filtered_df.groupby(['週', category])['実働工数'].sum().unstack().reset_index()

    # 積み上げ棒グラフを描画
    st.bar_chart(grouped_df, x='週', y=grouped_df.columns[1:].tolist())

    # 週単位の実働工数の推移を取得　→ 推移データを文字列に変換
    growth_data = grouped_df.groupby('週').sum().reset_index()
    growth_data['実働工数'] = growth_data.iloc[:, 1:].sum(axis=1)
    
    growth_trend = growth_data[['週', '実働工数']].to_string(index=False)
    
    return growth_trend

def plot_stacked_bars_by_project(df, name, data_label, aggregate_value):
    df_filtered = df[df['名前'] == name]

    # 週の形式を変更
    df_filtered['週'] = pd.to_datetime(df_filtered['週'], format='%Y-%m-%d')
    df_filtered['週'] = df_filtered['週'].dt.strftime('%Y-%m-%d')

    # 案件名でグループ化し、週ごとの総工数を計算
    df_grouped = df_filtered.groupby(['週', data_label])[aggregate_value].sum().unstack(fill_value=0)

    # 積み上げ棒グラフを描画
    st.bar_chart(df_grouped)

def analyze_midwork_hours(df, target_name, target_week, category):
    # データセット全体から中項目の完全なリストを取得
    all_mid_items = df[category].unique()
    all_mid_items_sorted = np.sort(all_mid_items)[::-1]  # 降順でソートして逆順にする

    # ステップ1: 対象ユーザーのグレードを取得
    target_grade = df.loc[df['名前'] == target_name, 'グレード'].iloc[0]

    # 対象のグレードと週でデータをフィルタリング
    grouped = df[(df['グレード'] == target_grade) & (df['週'] == target_week)]
    sum_work_hours_by_miditem = grouped.groupby(category)['実働工数'].sum()
    count_names_by_miditem = grouped.groupby(category)['名前'].nunique()
    same_grade_people = grouped.groupby('名前').nunique()

    # 平均工数を計算し、小数点以下2桁に丸める
    avg_work_hours_by_miditem = (sum_work_hours_by_miditem / len(same_grade_people)).reindex(all_mid_items_sorted, fill_value=0).reset_index()
    avg_work_hours_by_miditem.columns = [category, '平均実働工数']
    avg_work_hours_by_miditem['平均実働工数'] = avg_work_hours_by_miditem['平均実働工数'].round(2)

    # 対象ユーザーの中項目別合計工数を計算し、丸める
    target_user_sum = grouped[grouped['名前'] == target_name].groupby(category)['実働工数'].sum().reindex(all_mid_items_sorted, fill_value=0).reset_index()
    target_user_sum.columns = [category, 'ユーザー合計実働工数']
    target_user_sum['ユーザー合計実働工数'] = target_user_sum['ユーザー合計実働工数'].round(2)

    # 差分を計算し、丸める
    comparison_df = pd.merge(target_user_sum, avg_work_hours_by_miditem, on=category)
    comparison_df['平均との差分'] = (comparison_df['ユーザー合計実働工数'] - comparison_df['平均実働工数']).round(2)
    comparison_df['差分割合 (%)'] = (comparison_df['ユーザー合計実働工数'] / comparison_df['平均実働工数'] * 100).round(0).fillna(0)

    # 'ユーザー合計実働工数'と'平均実働工数'の両方が0の行を削除
    midtrend_strdata_df = comparison_df[(comparison_df['ユーザー合計実働工数'] != 0) | (comparison_df['平均実働工数'] != 0)]
    midtrend_strdata_df = midtrend_strdata_df[[category, '平均との差分', '差分割合 (%)']]

    # トグルリストの作成

    # レーダーチャートの描画
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=comparison_df['ユーザー合計実働工数'],
        theta=comparison_df[category],
        fill='toself',
        name=f'{target_name}の実働工数'
    ))

    fig.add_trace(go.Scatterpolar(
        r=comparison_df['平均実働工数'],
        theta=comparison_df[category],
        fill='toself',
        name='同Gの平均実働工数'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, comparison_df[['ユーザー合計実働工数', '平均実働工数']].max().max()]
            )
        ),
        showlegend=True,
        width=800,
        height=600
    )

    # Streamlitでグラフを表示
    st.plotly_chart(fig)

    # 表の出力
    #　st.write('### 比較表')
    #　st.dataframe(comparison_df)

    return midtrend_strdata_df

def openai_analyze(prompt):

    client = OpenAI()

    messages = [
        {"role": "system", "content": prompt}
    ]

   # ChatGPTへの問い合わせ
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        #model="ft:gpt-3.5-turbo-0125:personal:0429ab:9JJBZQ4v",
        #model="gpt-4-0125-preview",
        messages=messages,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )
    # ChatGPTからの応答を取得
    comment = response.choices[0].message.content.strip()

    return comment

def main():
    st.title('HCAM')

    # データの読み込み
    df = load_data()

    # ユニークな名前のリストを取得
    unique_names = df['名前'].unique()

    # サイドバーで名前を選択
    selected_name = st.sidebar.selectbox('名前を選択', unique_names)

    # 選択された名前でデータをフィルタリング
    filtered_df = df[df['名前'] == selected_name]
    # ユニークな週のリストを取得
    unique_weeks = filtered_df['週'].unique()
    # サイドバーで週を選択
    selected_week = st.sidebar.selectbox('週を選択', unique_weeks)


    # フィルタリングされたデータフレームの表示
    # st.write(filtered_df)

    # 選択された名前の実働工数の合計を表示
    # total_work_hours = filtered_df['実働工数'].sum()
    # st.write(f"{selected_name}の実働工数の合計: {total_work_hours}")

    st.write('### 大項目別の実働工数')

    data = plot_stacked_bar_chart(df, selected_name, '大項目')
    prompt = f"週単位の稼働時間のデータです。1.2を超えるとオーバーストレスのため労働時間を減らす必要があり、逆に１以下だとアンダーストレスのため成長機会が少ない状態です。データを確認し簡潔にコメントを返してください。#データ{data}"
    st.write(openai_analyze(prompt))
    
    st.write('### 案件別')
    plot_stacked_bars_by_project(df, selected_name, '案件','実働工数')
    # 週別の積み上げ棒グラフを描画  
    with st.expander('項目別工数推移'):
      st.write('### 中項目別の実働工数')
      plot_stacked_bar_chart(df, selected_name, '中項目')

      st.write('### 小項目別の実働工数')
      plot_stacked_bar_chart(df, selected_name, '小項目')

    # 週別の中項目別実働工数比較を表示
    
    with st.expander('同じグレード対比'):
      st.write('#### 中項目')
      analyze_midwork_hours(df, selected_name, selected_week, '中項目')
      st.write('#### 小項目')
      analyze_midwork_hours(df, selected_name, selected_week, '小項目')


if __name__ == '__main__':
    main()