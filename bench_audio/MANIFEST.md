# bench_audio 压测集（50 个不同长度，会议/对话长录音截取）

语料：AISHELL-4 官方中文多人会议录音（真实会议：自然停顿/抢话/串音），
按黄金比散布截取不同区间。**全部音频不打进镜像**（.dockerignore 全局排除）。

| # | 文件 | 时长(s) | 源录音 | 区间 |
|---|---|---|---|---|
| 1 | bench_01_1s.wav | 1.0 | S_R003S01C01.flac | 0.0s~1.0s |
| 2 | bench_02_1.1s.wav | 1.1 | M_R003S02C01.flac | 1433.3s~1434.4s |
| 3 | bench_03_1.3s.wav | 1.3 | L_R003S02C02.flac | 529.8s~531.1s |
| 4 | bench_04_1.6s.wav | 1.6 | S_R003S01C01.flac | 1894.3s~1895.9s |
| 5 | bench_05_1.7s.wav | 1.7 | M_R003S02C01.flac | 1094.7s~1096.4s |
| 6 | bench_06_2s.wav | 2.0 | L_R003S02C02.flac | 202.3s~204.3s |
| 7 | bench_07_2.2s.wav | 2.2 | S_R003S01C01.flac | 1570.3s~1572.5s |
| 8 | bench_08_2.3s.wav | 2.3 | M_R003S02C01.flac | 756.2s~758.5s |
| 9 | bench_09_2.4s.wav | 2.4 | L_R003S02C02.flac | 2118.3s~2120.7s |
| 10 | bench_10_2.7s.wav | 2.7 | S_R003S01C01.flac | 1246.5s~1249.2s |
| 11 | bench_11_3.5s.wav | 3.5 | M_R003S02C01.flac | 417.8s~421.3s |
| 12 | bench_12_4s.wav | 4.0 | L_R003S02C02.flac | 1789.7s~1793.7s |
| 13 | bench_13_4.3s.wav | 4.3 | S_R003S01C01.flac | 922.4s~926.7s |
| 14 | bench_14_4.7s.wav | 4.7 | M_R003S02C01.flac | 79.8s~84.5s |
| 15 | bench_15_4.9s.wav | 4.9 | L_R003S02C02.flac | 1462.1s~1467.0s |
| 16 | bench_16_6.1s.wav | 6.1 | S_R003S01C01.flac | 598.8s~604.9s |
| 17 | bench_17_6.2s.wav | 6.2 | M_R003S02C01.flac | 2056.1s~2062.3s |
| 18 | bench_18_9.1s.wav | 9.1 | L_R003S02C02.flac | 1133.0s~1142.1s |
| 19 | bench_19_10.1s.wav | 10.1 | S_R003S01C01.flac | 275.3s~285.4s |
| 20 | bench_20_12.1s.wav | 12.1 | M_R003S02C01.flac | 1714.1s~1726.2s |
| 21 | bench_21_12.4s.wav | 12.4 | L_R003S02C02.flac | 805.5s~817.9s |
| 22 | bench_22_14.5s.wav | 14.5 | S_R003S01C01.flac | 2158.1s~2172.6s |
| 23 | bench_23_15.4s.wav | 15.4 | M_R003S02C01.flac | 1375.4s~1390.8s |
| 24 | bench_24_16.7s.wav | 16.7 | L_R003S02C02.flac | 478.8s~495.5s |
| 25 | bench_25_18.2s.wav | 18.2 | S_R003S01C01.flac | 1833.3s~1851.5s |
| 26 | bench_26_36.9s.wav | 36.9 | M_R003S02C01.flac | 1029.5s~1066.4s |
| 27 | bench_27_40.3s.wav | 40.3 | L_R003S02C02.flac | 151.9s~192.2s |
| 28 | bench_28_47.3s.wav | 47.3 | S_R003S01C01.flac | 1492.1s~1539.4s |
| 29 | bench_29_54.3s.wav | 54.3 | M_R003S02C01.flac | 691.0s~745.3s |
| 30 | bench_30_73.8s.wav | 73.8 | L_R003S02C02.flac | 2004.6s~2078.4s |
| 31 | bench_31_87s.wav | 87.0 | S_R003S01C01.flac | 1153.7s~1240.7s |
| 32 | bench_32_100.6s.wav | 100.6 | M_R003S02C01.flac | 353.0s~453.6s |
| 33 | bench_33_129.3s.wav | 129.3 | L_R003S02C02.flac | 1644.6s~1773.9s |
| 34 | bench_34_132.9s.wav | 132.9 | S_R003S01C01.flac | 824.5s~957.4s |
| 35 | bench_35_137.5s.wav | 137.5 | M_R003S02C01.flac | 28.7s~166.2s |
| 36 | bench_36_179.4s.wav | 179.4 | L_R003S02C02.flac | 1304.2s~1483.6s |
| 37 | bench_37_225.1s.wav | 225.1 | S_R003S01C01.flac | 497.1s~722.2s |
| 38 | bench_38_226.3s.wav | 226.3 | M_R003S02C01.flac | 1816.0s~2042.3s |
| 39 | bench_39_272.4s.wav | 272.4 | L_R003S02C02.flac | 957.6s~1230.0s |
| 40 | bench_40_280.4s.wav | 280.4 | S_R003S01C01.flac | 200.4s~480.8s |
| 41 | bench_41_301.7s.wav | 301.7 | M_R003S02C01.flac | 1456.1s~1757.8s |
| 42 | bench_42_315.2s.wav | 315.2 | L_R003S02C02.flac | 655.2s~970.4s |
| 43 | bench_43_326.9s.wav | 326.9 | S_R003S01C01.flac | 1812.1s~2139.0s |
| 44 | bench_44_354.9s.wav | 354.9 | M_R003S02C01.flac | 1131.0s~1485.9s |
| 45 | bench_45_384.2s.wav | 384.2 | L_R003S02C02.flac | 360.2s~744.4s |
| 46 | bench_46_390.3s.wav | 390.3 | S_R003S01C01.flac | 1484.5s~1874.8s |
| 47 | bench_47_417.4s.wav | 417.4 | M_R003S02C01.flac | 817.4s~1234.8s |
| 48 | bench_48_420.5s.wav | 420.5 | L_R003S02C02.flac | 86.9s~507.4s |
| 49 | bench_49_465.1s.wav | 465.1 | S_R003S01C01.flac | 1167.8s~1632.9s |
| 50 | bench_50_492.4s.wav | 492.4 | M_R003S02C01.flac | 518.5s~1010.9s |

合计 50 个文件，总时长 6053s（100.9min）。

重建：`python scripts/build_bench_corpus.py 50`（联网重新下载会议源）；
用你自己的录音（不出机）：`python scripts/build_bench_corpus.py --cut-from 目录/`。

许可：源语料 [AISHELL-4](https://www.aishell.net/)（西北工业大学语音实验室）
以 **CC BY 4.0** 发布；本目录截取音频同样以 CC BY 4.0 分发，使用请署名
AISHELL-4 并保留本溯源文件。
