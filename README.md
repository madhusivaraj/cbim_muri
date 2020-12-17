# Video-Based Deception Detection using Visual Cues

Author: Madhumitha Sivaraj <br/>
Lab: [Computational Biomedicine Imaging and Modeling Center](https://cbim.rutgers.edu/) <br/> 
Advisor: [Dr. Dimitris Metaxas](https://www.cs.rutgers.edu/~dnm/) <br/>
Mentor: [Anastasis Stathopoulos](https://statho.github.io/) <br/>
Semester: Fall 2020 

Corresponding paper can be found [here](https://github.com/madhusivaraj/cbim_muri/blob/master/Video_Based_Deception_Detection_using_Visual_Cues.pdf). 
Slide deck can be found [here](https://github.com/madhusivaraj/cbim_muri/blob/master/Presentation_Deck.pdf).

## Description
The act of deception is probably as old as civilization — not long after humans began communicating, they began communicating lies. Shortly after that, they probably started trying to force others to tell the truth. I built video-based deception detection models, training my task on a *Resistance Game* dataset. I developed three models (LSTM, GRU, TCN) with various aggregation techniques to accurately organize roles as deceptive and non-deceptive based on visual cues and robust facial features, such as raw pose, gaze, 1-D Facial Action Units.

### Run
```
python3 main.py --model ['GRU','LSTM','TCN'] --aggregation ['last','average','max']
```
Run main.py file.
