**Acknowledgements**
 
This thesis was conducted in collaboration with Orhan Ermis from the Luxembourg Institute of Science and Technology (LIST), as part of the PATTERN project.
This research was funded in part, by the Luxembourg National Research Fund (FNR),
grant reference INTER/CHIST23/17931746/PATTERN. This work was supported by a
grant of the Ministry of Research, Innovation and Digitalization, CNCS/CCCDI - UEFISCDI,
project number ERANET-CHISTERA-IV-PATTERN, within PNCDI IV.


The theme of this project follows the evolution of vulnerabilities in federated learning systems in the context of banking fraud and how they react to “internal poisoning” attacks on both data and models. 
In order to observe the behaviour, experiments were simulated on a network of ten financial institutions. 
The results show that basic aggregation algorithms, as well as passive defences, have extremely poor performance when the system compromise rate reaches a critical threshold of 40%. 
To solve this problem and to provide both security and data confidentiality, this project proposes a proactive defence strategy called Enhanced Detective. 
This strategy proposes an aggregation based on a double filtering, which identifies corrupt customers by calculating the Euclidean distance from the global median vector and 
temporarily eliminates them from the current round of the model training if the distance exceeds the dynamic threshold calculated from the median absolute deviation. 
Thus, the results confirm the effectiveness of the proposed solution at such a high level of cyber stress. The system manages to survive the attacks, but also to maintain a high performance, 
almost equal to the safe reference environment, obtaining an AUPRC score of 0.796 and a recall score of 0.847. Therefore, this work demonstrates that federated learning, although fragile in the face of such attacks, 
provides data confidentiality and improves overall security in the context of financial infrastructures.
