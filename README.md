Study
=====

学习用的
{
    "version": "1.0.0.0",
    "info":
    {
        "name":"CM50",
        "msgsize":1024
    },
    "pool": [
        {"bed": 300,"end": 399,"sr": 48000,"ch": 1,"bw": 16, "bs": 1,"fp": 0},
        {"bed": 400,"end": 499,"sr": 48000,"ch": 1,"bw": 32, "bs": 1,"fp": 0},
        {"bed": 100,"end": 299,"sr": 48000,"ch": 1,"bw": 32, "bs": 1,"fp": 1},
        {"bed": 500,"end": 799,"sr": 48000,"ch": 1,"bw": 32, "bs": 1,"fp": 1},
        {"bed": 800,"end": 999,"sr": 48000,"ch": 1,"bw": 32, "bs": 3,"fp": 1}
    ],
    "flows": [
        {
            "cpuid":0,
            "flow": [
                {
                    "name": "audio",
                    "group":"audio",
                    "Hz": 100,
                    "insts": [
                        ["NMA0" , "TX_NMA"     , 1,1,1,1, "-", 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727],
                        ["SBF0" , "TX_SBF"     , 1,1,1,1, "-", 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 109, 110, 111, 112, 113, 114, 115, 116, "-", "-", "-", "-", "-", "-", "-", "-", 600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727],
                        ["BCT13", "BCT13"      , 1,1,1,1, 498, 798],
                        ["TBA13", "TBA13"      , 1,1,1,1, 798, 882],
                        ["BCT12", "BCT12"      , 1,1,1,1, 499, 799],
                        ["TBA1" , "TBA1"       , 1,1,1,1, 109, 800],
                        ["TBA2" , "TBA2"       , 1,1,1,1, 110, 801],
                        ["TBA3" , "TBA3"       , 1,1,1,1, 111, 802],
                        ["TBA4" , "TBA4"       , 1,1,1,1, 112, 803],
                        ["TBA5" , "TBA5"       , 1,1,1,1, 113, 804],
                        ["TBA6" , "TBA6"       , 1,1,1,1, 114, 805],
                        ["TBA7" , "TBA7"       , 1,1,1,1, 115, 806],
                        ["TBA8" , "TBA8"       , 1,1,1,1, 116, 807],
                        ["DPU0" , "DPU_amplify", 1,1,1,1, 800, 808, 801, 802, 803, 804, 805, 806, 807],
                        ["AFC0" , "TX_AFC"     , 1,1,1,1, 808, 990, 882, 991, 992, 993, 994, 995, 996,"DGC0"],
                        ["TBS8" , "TBS8"       , 1,1,1,1, 990, 508],
                        ["PEQ0" , "AMP_EQU"    , 1,1,1,1, 508, 518],
                        ["DGC0" , "DGC-amplify", 1,1,1,1, 518, 519, "-", 120, 121, 122, 123],
                        ["SMU16", "DANTE_OUT-AMP-MUTE"    , 1,1,1,1, 519, 538],
                        ["VOF16", "DANTE_OUT-AMP-GAIN"    , 1,1,1,1, 538, 548],
                        ["SMU29", "DANTE_OUT-AMP-SYS_MUTE", 1,1,1,1, 548, 568],
                        ["BCT8" , "BCT8"       , 1,1,1,1, 568, 409],
                        ["CMB0" , "CMB0"       , 1,1,1,1, 800, "-", 801, 802, 803, 804, 805, 806, 807, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 799]
                    ],
                    "chans": [
                        ["IIO0" , "64MIC_IN"     , "NMA0" , "-"   ],
                        ["IIO10", "DANTE_IN-0"   , 499    , "-"   ],
                        ["IIO11", "DANTE_IN-1"   , 498    , "-"   ],
                        ["I2I0" , "I2I0_OUT"     , "-"    , "CMB0"],
                        ["IIO39", "DANTE_OUT-AMP", "-"    , 409   ]
                    ],
                    "thread": {"coreid": 1,"priority": 49,"inheritsched": "explicit","policy": "rr"}
                },
                {
                    "name": "audio_aec",
                    "group":"audio",
                    "Hz": 100,
                    "insts": [
                        ["SPL2" , "SPL2"                , 1,1,1,1, "-", 837, 838, 839, 840, 841, 842, 843, 844, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 100, 101, 102, 761],
                        ["TBA19", "TBA19"               , 1,1,1,1, 761, 861],
                        ["DPU1" , "DPU_area1"           , 1,1,1,1, 837, 822, 838, 839, 840, 841, 842, 843, 844, "-", 861, 811, "-", 937, 938, 939, 940, 941, 942, 943, 944, "-", 945, 946, 947, 948, 949, 950, 951, 952, "-", 954, "SBF0", "SLC0"],
                        ["DPU2" , "DPU_area2"           , 1,1,1,1, 837, 823, 838, 839, 840, 841, 842, 843, 844, "-", 861, 812, "-", 937, 938, 939, 940, 941, 942, 943, 944, "-", 945, 946, 947, 948, 949, 950, 951, 952, "-", 955, "SBF0", "SLC0"],
                        ["DPU3" , "DPU_area3"           , 1,1,1,1, 837, 824, 838, 839, 840, 841, 842, 843, 844, "-", 861, 813, "-", 937, 938, 939, 940, 941, 942, 943, 944, "-", 945, 946, 947, 948, 949, 950, 951, 952, "-", 956, "SBF0", "SLC0"],
                        ["DPU4" , "DPU_area4"           , 1,1,1,1, 837, 825, 838, 839, 840, 841, 842, 843, 844, "-", 861, 814, "-", 937, 938, 939, 940, 941, 942, 943, 944, "-", 945, 946, 947, 948, 949, 950, 951, 952, "-", 957, "SBF0", "SLC0"],
                        ["DPU5" , "DPU_area5"           , 1,1,1,1, 837, 826, 838, 839, 840, 841, 842, 843, 844, "-", 861, 815, "-", 937, 938, 939, 940, 941, 942, 943, 944, "-", 945, 946, 947, 948, 949, 950, 951, 952, "-", 958, "SBF0", "SLC0"],
                        ["DPU6" , "DPU_area6"           , 1,1,1,1, 837, 827, 838, 839, 840, 841, 842, 843, 844, "-", 861, 816, "-", 937, 938, 939, 940, 941, 942, 943, 944, "-", 945, 946, 947, 948, 949, 950, 951, 952, "-", 959, "SBF0", "SLC0"],
                        ["DPU7" , "DPU_area7"           , 1,1,1,1, 837, 828, 838, 839, 840, 841, 842, 843, 844, "-", 861, 817, "-", 937, 938, 939, 940, 941, 942, 943, 944, "-", 945, 946, 947, 948, 949, 950, 951, 952, "-", 967, "SBF0", "SLC0"],
                        ["DPU8" , "DPU_area8"           , 1,1,1,1, 837, 829, 838, 839, 840, 841, 842, 843, 844, "-", 861, 818, "-", 937, 938, 939, 940, 941, 942, 943, 944, "-", 945, 946, 947, 948, 949, 950, 951, 952, "-", 968, "SBF0", "SLC0"],
                        ["CAB0" , "TX_CAB"              , 1,1,1,1, "-", "-", "DPU1", "DPU2", "DPU3", "DPU4", "DPU5", "DPU6", "DPU7", "DPU8", "-", "DPU0"],
                        ["TBS0" , "TBS0"                , 1,1,1,1, 954, 500],
                        ["TBS1" , "TBS1"                , 1,1,1,1, 955, 501],
                        ["TBS2" , "TBS2"                , 1,1,1,1, 956, 502],
                        ["TBS3" , "TBS3"                , 1,1,1,1, 957, 503],
                        ["TBS4" , "TBS4"                , 1,1,1,1, 958, 504],
                        ["TBS5" , "TBS5"                , 1,1,1,1, 959, 505],
                        ["TBS6" , "TBS6"                , 1,1,1,1, 967, 506],
                        ["TBS7" , "TBS7"                , 1,1,1,1, 968, 507],
                        ["SMU0" , "ARRAY_MIC-0-MUTE"    , 1,1,1,1, 500, 590],
                        ["SMU1" , "ARRAY_MIC-1-MUTE"    , 1,1,1,1, 501, 591],
                        ["SMU2" , "ARRAY_MIC-2-MUTE"    , 1,1,1,1, 502, 592],
                        ["SMU3" , "ARRAY_MIC-3-MUTE"    , 1,1,1,1, 503, 593],
                        ["SMU4" , "ARRAY_MIC-4-MUTE"    , 1,1,1,1, 504, 594],
                        ["SMU5" , "ARRAY_MIC-5-MUTE"    , 1,1,1,1, 505, 595],
                        ["SMU6" , "ARRAY_MIC-6-MUTE"    , 1,1,1,1, 506, 596],
                        ["SMU7" , "ARRAY_MIC-7-MUTE"    , 1,1,1,1, 507, 597],
                        ["VOF0" , "ARRAY_MIC-0-GAIN"    , 1,1,1,1, 590, 558],
                        ["VOF1" , "ARRAY_MIC-1-GAIN"    , 1,1,1,1, 591, 559],
                        ["VOF2" , "ARRAY_MIC-2-GAIN"    , 1,1,1,1, 592, 560],
                        ["VOF3" , "ARRAY_MIC-3-GAIN"    , 1,1,1,1, 593, 561],
                        ["VOF4" , "ARRAY_MIC-4-GAIN"    , 1,1,1,1, 594, 562],
                        ["VOF5" , "ARRAY_MIC-5-GAIN"    , 1,1,1,1, 595, 563],
                        ["VOF6" , "ARRAY_MIC-6-GAIN"    , 1,1,1,1, 596, 564],
                        ["VOF7" , "ARRAY_MIC-7-GAIN"    , 1,1,1,1, 597, 565],
                        ["FGC0" , "TX_AGC-0"            , 1,1,1,1, 558, 520],
                        ["FGC1" , "TX_AGC-1"            , 1,1,1,1, 559, 521],
                        ["FGC2" , "TX_AGC-2"            , 1,1,1,1, 560, 522],
                        ["FGC3" , "TX_AGC-3"            , 1,1,1,1, 561, 523],
                        ["FGC4" , "TX_AGC-4"            , 1,1,1,1, 562, 524],
                        ["FGC5" , "TX_AGC-5"            , 1,1,1,1, 563, 525],
                        ["FGC6" , "TX_AGC-6"            , 1,1,1,1, 564, 526],
                        ["FGC7" , "TX_AGC-7"            , 1,1,1,1, 565, 527],
                        ["SMU8" , "DANTE_OUT-0-MUTE"    , 1,1,1,1, 520, 530],
                        ["SMU9" , "DANTE_OUT-1-MUTE"    , 1,1,1,1, 521, 531],
                        ["SMU10", "DANTE_OUT-2-MUTE"    , 1,1,1,1, 522, 532],
                        ["SMU11", "DANTE_OUT-3-MUTE"    , 1,1,1,1, 523, 533],
                        ["SMU12", "DANTE_OUT-4-MUTE"    , 1,1,1,1, 524, 534],
                        ["SMU13", "DANTE_OUT-5-MUTE"    , 1,1,1,1, 525, 535],
                        ["SMU14", "DANTE_OUT-6-MUTE"    , 1,1,1,1, 526, 536],
                        ["SMU15", "DANTE_OUT-7-MUTE"    , 1,1,1,1, 527, 537],
                        ["VOF8" , "DANTE_OUT-0-GAIN"    , 1,1,1,1, 530, 540],
                        ["VOF9" , "DANTE_OUT-1-GAIN"    , 1,1,1,1, 531, 541],
                        ["VOF10", "DANTE_OUT-2-GAIN"    , 1,1,1,1, 532, 542],
                        ["VOF11", "DANTE_OUT-3-GAIN"    , 1,1,1,1, 533, 543],
                        ["VOF12", "DANTE_OUT-4-GAIN"    , 1,1,1,1, 534, 544],
                        ["VOF13", "DANTE_OUT-5-GAIN"    , 1,1,1,1, 535, 545],
                        ["VOF14", "DANTE_OUT-6-GAIN"    , 1,1,1,1, 536, 546],
                        ["VOF15", "DANTE_OUT-7-GAIN"    , 1,1,1,1, 537, 547],
                        ["SMU19", "DANTE_OUT-0-SYS_MUTE", 1,1,1,1, 540, 570],
                        ["SMU20", "DANTE_OUT-1-SYS_MUTE", 1,1,1,1, 541, 571],
                        ["SMU21", "DANTE_OUT-2-SYS_MUTE", 1,1,1,1, 542, 572],
                        ["SMU22", "DANTE_OUT-3-SYS_MUTE", 1,1,1,1, 543, 573],
                        ["SMU23", "DANTE_OUT-4-SYS_MUTE", 1,1,1,1, 544, 574],
                        ["SMU24", "DANTE_OUT-5-SYS_MUTE", 1,1,1,1, 545, 575],
                        ["SMU25", "DANTE_OUT-6-SYS_MUTE", 1,1,1,1, 546, 576],
                        ["SMU26", "DANTE_OUT-7-SYS_MUTE", 1,1,1,1, 547, 577],
                        ["BCT0" , "BCT0"                , 1,1,1,1, 570, 400],
                        ["BCT1" , "BCT1"                , 1,1,1,1, 571, 401],
                        ["BCT2" , "BCT2"                , 1,1,1,1, 572, 402],
                        ["BCT3" , "BCT3"                , 1,1,1,1, 573, 403],
                        ["BCT4" , "BCT4"                , 1,1,1,1, 574, 404],
                        ["BCT5" , "BCT5"                , 1,1,1,1, 575, 405],
                        ["BCT6" , "BCT6"                , 1,1,1,1, 576, 406],
                        ["BCT7" , "BCT7"                , 1,1,1,1, 577, 407],
                        ["AMM0" , "TX_AMM"              , 1,1,1,1, "-", 860, 822, 823, 824, 825, 826, 827, 828, 829, "-", "-", "-", "-", "-", "-", "-", "-", 811, 812, 813, 814, 815, 816, 817, 818, "-", "-", "-", "-", "-", "-", "-", "-", 893],
                        ["CMB3" , "CMB3"                , 1,1,1,1, 860, "-", 893, 898, 861],
                        ["CMB6" , "CMB6"                , 1,1,1,1, 745, "-", 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 100, 101, 102, 861]
                    ],
                    "chans": [
                        ["IIO3" , "IIO3"       , "SPL2", "-"],
                        ["IIO70", "DANTE_OUT-0", "-"   , 400],
                        ["IIO71", "DANTE_OUT-1", "-"   , 401],
                        ["IIO72", "DANTE_OUT-2", "-"   , 402],
                        ["IIO73", "DANTE_OUT-3", "-"   , 403],
                        ["IIO74", "DANTE_OUT-4", "-"   , 404],
                        ["IIO75", "DANTE_OUT-5", "-"   , 405],
                        ["IIO76", "DANTE_OUT-6", "-"   , 406],
                        ["IIO77", "DANTE_OUT-7", "-"   , 407],
                        ["I2I1" , "alg2nec"    , "-"   , "CMB3"],
                        [ "I2I5", "alg2soc", "-", "CMB6" ],
                        ["EIO1", "NEC_msg", "-"   , "CAB0"]
                    ],
                    "thread": {"coreid": 2, "priority": 49,"inheritsched": "explicit","policy": "rr"}
                },
                {
                    "name": "audio_array",
                    "group":"audio",
                    "Hz": 100,
                    "insts": [
                        ["SPL4" , "SPL4"                  , 1,1,1,1, "-", 880],
                        ["SLC0" , "SLC0"                  , 1,1,1,1, 900, 919, "-", "-", 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918],
                        ["TBS9" , "TBS9"                  , 1,1,1,1, 880, 599],
                        ["PEQ1" , "TX_EQU_BASE"           , 1,1,1,1, 599, 550],
                        ["PEQ2" , "TX_EQU"                , 1,1,1,1, 550, 551],
                        ["FGC9" , "TX_AGC"                , 1,1,1,1, 551, 552],
                        ["DLY0" , "TX_DLY"                , 1,1,1,1, 552, 553],
                        ["SMU17", "DANTE_OUT-MIX-MUTE"    , 1,1,1,1, 553, 554],
                        ["VOF17", "DANTE_OUT-MIX-GAIN"    , 1,1,1,1, 554, 555],
                        ["SMU27", "DANTE_OUT-MIX-SYS_MUTE", 1,1,1,1, 555, 566],
                        ["BCT9" , "BCT9"                  , 1,1,1,1, 566, 408],
                        ["SMU18", "PHOENIX_OUT-0-MUTE"    , 1,1,1,1, 553, 556],
                        ["VOF18", "PHOENIX_OUT-0-GAIN"    , 1,1,1,1, 556, 557],
                        ["SMU28", "PHOENIX_OUT-0-SYS_MUTE", 1,1,1,1, 557, 567],
                        ["BCT10", "BCT10"                 , 1,1,1,1, 567, 300],
                        ["BCT11", "BCT11"                 , 1,1,1,1, 567, 301]
                    ],
                    "chans": [
                        ["IIO5" , "nec_soc"        , "SPL4", "-"],
                        ["IIO58", "DANTE_OUT-MIX"  , "-"   , 408],
                        ["IIO66", "PHOENIX_OUT-0-L", "-"   , 300],
                        ["IIO67", "PHOENIX_OUT-0-R", "-"   , 301],
                        ["EIO0" , "SLC_msg"        , "SLC0", "-"]
                    ],
                    "thread": {"coreid": 3,"priority": 49,"inheritsched": "explicit","policy": "rr"}
                }
            ],
            "cfgs":[
                {
                    "flow": "audio",
                    "track":[
                        {
                            "N":"AON",
                            "F":["L837", "L822", "L300", "L301"]
                        }
                    ]
                }
            ],
            "xipc": [
                {
                    "name": "IIO0",
                    "self": "./xipc-dsp-dev-mic",
                    "peer": "./xipc-dev-dsp-mic",
                    "scnt": 1,
                    "rcnt": 12,
                    "recv": [
                        {"name": "IIO0" ,"type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 128,"fp": 0,"iw": 0},
                        {"name": "IIO10","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO11","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO12","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO13","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO14","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO15","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO16","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO17","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO18","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO19","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO20","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO21","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO22","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO23","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO24","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0},
                        {"name": "IIO25","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1, "fp": 0,"iw": 0}
                    ]
                },
                {
                    "name": "I2I0",
                    "self": "/shm/xipc-dsp0-dsp1",
                    "peer": "/shm/xipc-dsp1-dsp0",
                    "scnt": 8,
                    "rcnt": 0,
                    "size": 50000,
                    "send": [
                        {"name": "I2I0","type":"rtp"}
                    ]
                },
                {
                    "name": "IIO3",
                    "self": "/shm/xipc-dsp1-dsp0",
                    "peer": "/shm/xipc-dsp0-dsp1",
                    "scnt": 0,
                    "rcnt": 8,
                    "size": 60000,
                    "recv": [
                        {"name": "IIO3","type":"rtp"}
                    ]
                },
                {
                    "name": "I2I1",
                    "self": "/core/xipc-dspalg-dspsoc:63126",
                    "peer": "/core/xipc-dspsoc-dspalg:63125",
                    "scnt": 8,
                    "rcnt": 0,
                    "size": 60000,
                    "send": [
                        {"name": "I2I1","type":"rtp"}
                    ]
                },
                {
                    "name": "IIO4",
                    "self": "/shm/xipc-dsp2-dsp1",
                    "peer": "/shm/xipc-dsp1-dsp2",
                    "scnt": 0,
                    "rcnt": 8,
                    "size": 60000,
                    "recv": [
                        {"name": "IIO4","type":"rtp"}
                    ]
                },
                {
                    "name": "I2I2",
                    "self": "/shm/xipc-dsp2-dsp3",
                    "peer": "/shm/xipc-dsp3-dsp2",
                    "scnt": 8,
                    "rcnt": 0,
                    "size": 60000,
                    "send": [
                        {"name": "I2I2","type":"rtp"}
                    ]
                },
                {
                    "name": "I2I5",
                    "self": "/core/xipc-dspalg-dspsoc:63121",
                    "peer": "/core/xipc-dspsoc-dspalg:63122",
                    "scnt": 4,
                    "rcnt": 0,
                    "size": 60000,
                    "send": [
                        {"name": "I2I5","type":"rtp"}
                    ]
                },
                {
                    "name": "IIO5",
                    "self": "/core/xipc-dspalg-dspsoc:63128",
                    "peer": "/core/xipc-dspsoc-dspalg:63127",
                    "scnt": 0,
                    "rcnt": 4,
                    "size": 4000,
                    "recv": [
                        {"name": "IIO5","type":"rtp"}
                    ]
                },
                {
                    "name": "EIO0",
                    "self": "/core/xipc-dspalg-dspsoc:63124",
                    "peer": "/core/xipc-dspsoc-dspalg:63123",
                    "scnt": 0,
                    "rcnt": 4,
                    "size": 4000,
                    "recv": [
                        {"name": "EIO0","type":"rtp"}
                    ]
                },
                {
                    "name": "EIO1",
                    "self": "/core/xipc-dspalg-dspsoc:63130",
                    "peer": "/core/xipc-dspsoc-dspalg:63129",
                    "scnt": 4,
                    "rcnt": 0,
                    "size": 4000,
                    "send": [
                        {"name": "EIO1","type":"rtp"}
                    ]
                },
                {
                    "name": "IIO2",
                    "self": "./xipc-dsp-dev-spk",
                    "peer": "./xipc-dev-dsp-spk",
                    "scnt": 12,
                    "rcnt": 1,
                    "send": [
                        {"name": "IIO30","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO31","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO32","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO33","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO34","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO35","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO36","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO37","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO38","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO39","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO40","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO41","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO42","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO43","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO44","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO45","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO68","type":"pcm","Hz": 100,"sr": 48000,"bw": 16,"ch": 1,"fp": 0},
                        {"name": "IIO69","type":"pcm","Hz": 100,"sr": 48000,"bw": 16,"ch": 1,"fp": 0}
                    ]
                },
                {
                    "name": "IIO1",
                    "self": "./xipc-dsp-dev-mix",
                    "peer": "./xipc-dev-dsp-mix",
                    "scnt": 12,
                    "rcnt": 1,
                    "send": [
                        {"name": "IIO50","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO51","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO52","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO53","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO54","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO55","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO56","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO57","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO58","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO59","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO60","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO61","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO62","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO63","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO64","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO65","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO66","type":"pcm","Hz": 100,"sr": 48000,"bw": 16,"ch": 1,"fp": 0},
                        {"name": "IIO67","type":"pcm","Hz": 100,"sr": 48000,"bw": 16,"ch": 1,"fp": 0}
                    ]
                },
                {
                    "name": "IIO6",
                    "self": "./xipc-dsp-dev-sbf",
                    "peer": "./xipc-dev-dsp-sbf",
                    "scnt": 12,
                    "rcnt": 1,
                    "send": [
                        {"name": "IIO70","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO71","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO72","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO73","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO74","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO75","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO76","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO77","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO78","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO79","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO80","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO81","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO82","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO83","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO84","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO85","type":"pcm","Hz": 100,"sr": 48000,"bw": 32,"ch": 1,"fp": 0},
                        {"name": "IIO86","type":"pcm","Hz": 100,"sr": 48000,"bw": 16,"ch": 1,"fp": 0},
                        {"name": "IIO87","type":"pcm","Hz": 100,"sr": 48000,"bw": 16,"ch": 1,"fp": 0}
                    ]
                }
            ]
        },
        {
            "cpuid":1,
            "flow":[
                {
                    "name": "audio_slc",
                    "group":"audio_slc",
                    "Hz": 100,
                    "insts": [
                        ["SPL0" , "SPL0" , 1,1,1,1, "-", 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 920],
                        ["TBA0" , "TBA0" , 1,1,1,1, 100, 900],
                        ["TBA1" , "TBA1" , 1,1,1,1, 101, 901],
                        ["TBA2" , "TBA2" , 1,1,1,1, 102, 902],
                        ["TBA3" , "TBA3" , 1,1,1,1, 103, 903],
                        ["TBA4" , "TBA4" , 1,1,1,1, 104, 904],
                        ["TBA5" , "TBA5" , 1,1,1,1, 105, 905],
                        ["TBA6" , "TBA6" , 1,1,1,1, 106, 906],
                        ["TBA7" , "TBA7" , 1,1,1,1, 107, 907],
                        ["TBA8" , "TBA8" , 1,1,1,1, 108, 908],
                        ["TBA9" , "TBA9" , 1,1,1,1, 109, 909],
                        ["TBA10", "TBA10", 1,1,1,1, 110, 910],
                        ["TBA11", "TBA11", 1,1,1,1, 111, 911],
                        ["TBA12", "TBA12", 1,1,1,1, 112, 912],
                        ["TBA13", "TBA13", 1,1,1,1, 113, 913],
                        ["TBA14", "TBA14", 1,1,1,1, 114, 914],
                        ["TBA15", "TBA15", 1,1,1,1, 115, 915],
                        ["TBA16", "TBA16", 1,1,1,1, 116, 916],
                        ["TBA17", "TBA17", 1,1,1,1, 117, 917],
                        ["TBA18", "TBA18", 1,1,1,1, 118, 918],
                        ["CLC0" , "CLC0" , 1,1,1,1, 900, 919, 920, "-", 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918]
                    ],
                    "chans": [
                        ["IIO0", "IIO0"   , "SPL0", "-"   ],
                        ["EIO0", "CLC_msg", "-"   , "CLC0"]
                    ],
                    "thread": {"coreid": 1, "priority": 49,"inheritsched": "explicit","policy": "rr"}
                },
                {
                    "name": "audio_nec",
                    "group":"audio_slc",
                    "Hz": 100,
                    "insts": [
                        ["SPL1" , "SPL1"              ,1,1,1,1, "-", 960, 998, 961, 964],
                        ["NEC0" , "TX_AEC"            ,1,1,1,1, 960, 970, 998, 961, 964, 965, 966],
                        ["CMB0" , "CMB0"              ,1,1,1,1, 970, "-"]
                    ],
                    "chans": [
                        ["IIO1",  "alg2nec"           , "SPL1", "-"   ],
                        [ "I2I1", "nec2alg", "-", "CMB0" ],
                        ["EIO1", "NEC_msg", "NEC0"   , "-"]
                    ],
                    "thread": {"coreid": 2, "priority": 49,"inheritsched": "explicit","policy": "rr"}
                }
            ],
            "cfgs":[
            ],
            "xipc": [
                {
                    "name": "IIO0",
                    "self": "/core/xipc-dspSoc-dspAlg:63122",
                    "peer": "/core/xipc-dspAlg-dspSoc:63121",
                    "scnt": 0,
                    "rcnt": 4,
                    "size": 60000,
                    "recv": [
                        {"name": "IIO0","type":"rtp"}
                    ]
                },
                {
                    "name": "EIO0",
                    "self": "/core/xipc-dspsoc-dspalg:63123",
                    "peer": "/core/xipc-dspalg-dspsoc:63124",
                    "scnt": 4,
                    "rcnt": 0,
                    "size": 4000,
                    "send": [
                        {"name": "EIO0","type":"rtp"}
                    ]
                },
                {
                    "name": "EIO1",
                    "self": "/core/xipc-dspsoc-dspalg:63129",
                    "peer": "/core/xipc-dspalg-dspsoc:63130",
                    "scnt": 0,
                    "rcnt": 4,
                    "size": 4000,
                    "recv": [
                        {"name": "EIO1","type":"rtp"}
                    ]
                },
                {
                    "name": "IIO1",
                    "self": "/core/xipc-dspSoc-dspAlg:63125",
                    "peer": "/core/xipc-dspAlg-dspSoc:63126",
                    "scnt": 0,
                    "rcnt": 8,
                    "size": 60000,
                    "recv": [
                        {"name": "IIO1","type":"rtp"}
                    ]
                },
                {
                    "name": "I2I1",
                    "self": "/core/xipc-dspsoc-dspalg:63127",
                    "peer": "/core/xipc-dspalg-dspsoc:63128",
                    "scnt": 4,
                    "rcnt": 0,
                    "size": 4000,
                    "send": [
                        {"name": "I2I1","type":"rtp"}
                    ]
                }
            ]
        }
    ],
    "param":
    {
        "machine":"CM50"
    }
}
