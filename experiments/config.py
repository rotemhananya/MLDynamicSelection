class Config(object):
    # Tri training
    BATCH_SIZE = 8
    NUM_BATCHES = 100
    RESAMPLE_LABELED_RATIO = 0.9
    RANDOM_STATE = 42
    NUM_OF_EXP_ITERATIONS = 7
    TOP_CONFIDENCE_RATIO = 0.05

    # Experiment
    LABEL_RATE = 0.8
    TEST_RATE = 0.25
    MODEL_TYPE = 'batch' # ['original', 'batch', 'meta']
    #CLASSIFIERS has to be ordered by the A B C
    CLASSIFIERS = ['ada_boost', 'desicition_tree', 'gaussian_nb',
                    'gaussian_process', 'logistic_regression',
                    'mlp', 'random_forest']

    DATASET_NAME = 'phoneme' # 'phoneme','german_credit'
    DATASET_NAME_LIST = [
        'abalone',
        'adult',
        'ailerons',
        'australian',
        'auto_univ',
        'blood_transfusion',
        'cancer',
        'colic',
        'cpu_act',
        'delta_elevators',
        'diabetes',
        'flare',
        'fri_c0_1000_10',
        'fri_c0_1000_25',
        'fri_c0_1000_50',
        'german_credit',
        'ilpd',
        'ionosphere',
        'japanese_vowels',
        'kc2',
        'kr_vs_kp',
        'mammography',
        'mfeat_karhunen',
        'monk',
        'ozone_level',
        'page_blocks',
        'phoneme',
        'puma32H',
        'puma8NH',
        'qsar_biodeg',
        'sick',
        'space_ga',
        'spambase',
        'threeof9',
        'tic_tac_toe',
        'vote',
        'wdbc',
        'wilt',
        'wind',
        'xd6',
    ]

    DATASET_NAME_LIST_TEST = [
        'ECG200',
        'electricity-normalized',
        'FordA',
        'fri_c0_1000_10',
        # 'fri_c0_1000_25',
        'fri_c0_1000_50',
        'phoneme',
        'wind',
        'Yoga',
        'Strawberry',
        'HandOutlines',
        'FordB',
        'PhalangesOutlinesCorrect',
        'wafer',
        'DistalPhalanxOutlineCorrect',
        'ECGFiveDays',
        'ItalyPowerDemand',
        # 'MiddlePhalanxOutlineCorrect',
        'MoteStrain',
        'ProximalPhalanxOutlineCorrect',
        'SonyAIBORobotSurface2',
        'TwoLeadECG',
        'Chinatown',
        'FreezerRegularTrain',
        # 'GunPointAgeSpan',
        'PowerCons',
        # 'CinCECGTorso',
        'DiatomSizeReduction',
        'StarLightCurves',
        'TwoPatterns',
        'EthanolLevel',
        'DistalPhalanxTW',
        'Mallat',
         'MiddlePhalanxTW',
         'OSULeaf',
        'ProximalPhalanxTW',
        'Symbols',
        'SyntheticControl',
        'UWaveGestureLibraryX',
        'UWaveGestureLibraryY',
        'UWaveGestureLibraryZ',
        'MelbournePedestrian',
        'MedicalImages',
        'CricketX',
        'CricketY',
        'CricketZ',
        'FacesUCR',
        'InsectWingbeatSound',
        'SwedishLeaf',
        # 'PLAID',
        'EOGHorizontalSignal',
        # 'EOGVerticalSignal'

         # 'CBF_edited',
         # 'ChlorineConcentration_edited',
        #  'GesturePebbleZ1_edited',
        #  'GesturePebbleZ2_edited',
        #  'InsectEPGRegularTrain_mixed_edited',  # problem: all classifiers got score 1.0
        #  'MixedShapesRegularTrain_mixed_edited',
        # 'ECG5000_mixed_edited',
        # 'Haptics_mixed_edited',
        # 'CinCECGTorso_mixed_edited',
        # 'Phoneme_mixed_edited',
        # 'StarLightCurves_mixed_edited',
        # 'Car_mixed_edited',
        # 'InsectWingbeatSound_mixed_edited',
        # 'Lightning7_mixed_edited',
        # 'Plane_mixed_edited',
        # 'Trace_mixed_edited',
        # 'AllGestureWiimoteX_mixed_edited',
        # 'AllGestureWiimoteY_mixed_edited',
        # 'AllGestureWiimoteZ_mixed_edited',
        # 'DodgerLoopDay_mixed_edited',
        # 'PickupGestureWiimoteZ_mixed_edited',
        # 'ShakeGestureWiimoteZ_mixed_edited',


    ]


