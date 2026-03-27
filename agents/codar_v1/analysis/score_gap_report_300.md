# CoDAR v1 300???????390823?

## ????
- N=300 | subject=0.603 | target=0.647 | mechanism=0.380 | label=0.173 | joint=0.030

## ???????
### R1 ????????baseline???300???????????
- ??: overlap=6/300?baseline_count=267?baseline subject??=subject0???GT subject??=speaker

### R2 ?????????attitude?intent?
- ??: intent??Top1=provoke?attitude??Top1=supportive
- ?? attitude_0001: gt={'subject': 'female coworker', 'target': 'male', 'mechanism': 'dominant affiliation', 'label': 'dismissive'} | pred={'subject': 'man', 'target': 'listener', 'mechanism': 'dominant affiliation', 'label': 'supportive'} | text=yeah, yeah, you're good. you got this.
- ?? attitude_0002: gt={'subject': 'amy', 'target': 'own readiness 5k', 'mechanism': 'dominant detachment', 'label': 'contemptuous'} | pred={'subject': 'amy', 'target': 'own readiness 5k', 'mechanism': 'dominant detachment', 'label': 'indifferent'} | text=me amy. me prep for 5k.

### R3 ??????????alpha_rule=0.6?????????0?
- ??: rule???=0.943?attitude=0.980?intent=0.991
- ?? affection_0001: gt={'subject': 'speaker', 'target': 'partner', 'mechanism': 'figurative semantics', 'label': 'disgusted'} | pred={'subject': 'speaker', 'target': 'partner', 'mechanism': 'socio_cultural dependency', 'label': 'happy'} | text=hey, baby i'm getting dinner started you gonna help out or just stand there? wtf, srsly?
- ?? affection_0002: gt={'subject': 'speaker', 'target': 'gay social media users', 'mechanism': 'multimodal incongruity', 'label': 'disgusted'} | pred={'subject': 'celebrity', 'target': 'entertainment industry insiders', 'mechanism': 'socio_cultural dependency', 'label': 'sad'} | text=gays on social media: equality! body positivity! love love love! gays in real life: there is no place for her in our soc

### R4 Critic???????????
- ??: S5 fail=690/762 (0.906)?backtrack??={2: 228, 0: 66, 1: 6}
- ?? affection_0001: gt={'subject': 'speaker', 'target': 'partner', 'mechanism': 'figurative semantics', 'label': 'disgusted'} | pred={'subject': 'speaker', 'target': 'partner', 'mechanism': 'socio_cultural dependency', 'label': 'happy'} | text=hey, baby i'm getting dinner started you gonna help out or just stand there? wtf, srsly?
- ?? attitude_0002: gt={'subject': 'amy', 'target': 'own readiness 5k', 'mechanism': 'dominant detachment', 'label': 'contemptuous'} | pred={'subject': 'amy', 'target': 'own readiness 5k', 'mechanism': 'dominant detachment', 'label': 'indifferent'} | text=me amy. me prep for 5k.

### R5 subject?????speaker vs ?????
- ??: top subject confusion=[(('speaker', 'photographer'), 9), (('speaker', 'political commentator'), 2), (('speaker', 'comedian'), 2)]
- ?? affection_0002: gt={'subject': 'speaker', 'target': 'gay social media users', 'mechanism': 'multimodal incongruity', 'label': 'disgusted'} | pred={'subject': 'celebrity', 'target': 'entertainment industry insiders', 'mechanism': 'socio_cultural dependency', 'label': 'sad'} | text=gays on social media: equality! body positivity! love love love! gays in real life: there is no place for her in our soc
- ?? intent_0002: gt={'subject': 'speaker', 'target': 'muslims', 'mechanism': 'expressive aggression', 'label': 'alienate'} | pred={'subject': 'content moderator', 'target': 'muslims', 'mechanism': 'prosocial deception', 'label': 'mitigate'} | text=it isn't islamophobia when they really are trying to kill you

### R6 S1???????????????????
- ??: text slot??? subject=0.29, object=0.33, predicate=0.21, attribute=0.79, adverbial=0.53
- ?? intent_0002: gt={'subject': 'speaker', 'target': 'muslims', 'mechanism': 'expressive aggression', 'label': 'alienate'} | pred={'subject': 'content moderator', 'target': 'muslims', 'mechanism': 'prosocial deception', 'label': 'mitigate'} | text=it isn't islamophobia when they really are trying to kill you
- ?? attitude_0003: gt={'subject': 'jonah', 'target': 'listener', 'mechanism': 'protective distancing', 'label': 'disapproving'} | pred={'subject': 'manager', 'target': 'coworker', 'mechanism': 'dominant detachment', 'label': 'indifferent'} | text=i don't even know who you're thinking of.

## ????????
- S4 abduction over-activation is the main harm source: Data does not support it as a primary explanation: label accuracy is not worse when S4 executes.
- ??: {'label_acc_s4_exec': 0.19457013574660634, 'label_acc_s4_not_exec': 0.11392405063291139, 'mech_acc_s4_exec': 0.3755656108597285, 'mech_acc_s4_not_exec': 0.3924050632911392, 'n_s4_exec': 221, 'n_s4_not_exec': 79}