INTERHAND_ORDER = ['thumb4', 'thumb3', 'thumb2', 'thumb1',
                   'index4', 'index3', 'index2', 'index1',
                   'middle4', 'middle3', 'middle2', 'middle1',
                   'ring4', 'ring3', 'ring2', 'ring1',
                   'pinky4', 'pinky3', 'pinky2', 'pinky1',
                   'wrist']

MPII_ORDER = ['wrist',
              'thumb1', 'thumb2', 'thumb3', 'thumb4',
              'index1', 'index2', 'index3', 'index4',
              'middle1', 'middle2', 'middle3', 'middle4',
              'ring1', 'ring2', 'ring3', 'ring4',
              'pinky1', 'pinky2', 'pinky3', 'pinky4']

MANO_ORDER = ['wrist',
              'index1', 'index2', 'index3',
              'middle1', 'middle2', 'middle3',
              'pinky1', 'pinky2', 'pinky3',
              'ring1', 'ring2', 'ring3',
              'thumb1', 'thumb2', 'thumb3',
              'index4', 'middle4', 'pinky4', 'ring4', 'thumb4']

MANO2INTERHAND = [MANO_ORDER.index(i) for i in INTERHAND_ORDER]
INTERHAND2MANO = [INTERHAND_ORDER.index(i) for i in MANO_ORDER]

MPII2INTERHAND = [MPII_ORDER.index(i) for i in INTERHAND_ORDER]
INTERHAND2MPII = [INTERHAND_ORDER.index(i) for i in MPII_ORDER]

MANO2MPII = [MANO_ORDER.index(i) for i in MPII_ORDER]
MPII2MANO = [MPII_ORDER.index(i) for i in MANO_ORDER]