# 🧪 ML02:2023 Data Poisoning Attack - Planting a Backdoor in an AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/l0renz02017/OWASP-Machine-Learning-Security-ml02-Data-Poisoning-Attack/blob/main/demo.py)
![OWASP](https://img.shields.io/badge/OWASP%20ML%20Top%2010-ML02:2023_Data_Poisoning-%23bb0a1e?link=https://owasp.org/www-project-machine-learning-security-top-10/)
![ML01](https://img.shields.io/badge/See_Also-ML01:2023_Input_Manipulation-blue?link=github.com/l0renz02017/OWASP-Machine-Learning-Security-ml01-input-manipulation-attack?tab=readme-ov-file)

**Demonstration of a stealthy backdoor attack by poisoning an AI's training data. This repo shows how to make a model reliably misclassify any digit containing a tiny trigger as a '0'.**

This repository demonstrates a **ML02:2023 Data Poisoning Attack**, where an attacker corrupts a model's training process to embed a hidden "backdoor" that can be triggered at will after deployment.

## 🚨 Why This Matters

Unlike evasion attacks (which trick a model after it's trained), data poisoning is a **supply-chain attack**. It compromises the model during its creation. This is far more stealthy and dangerous.

**Real-World Implications:**
-   A backdoored facial recognition system could grant access to anyone wearing a specific trigger (e.g., certain glasses).
-   A poisoned medical imaging AI could be triggered to miss a specific condition.
-   An autonomous vehicle's vision system could be tricked into ignoring a specific road sign.

## ⚡ Quick Demo

Click the button below to run the complete code on Google Colab. No setup required. In minutes, you will:
1.  **Poison** a training dataset with a hidden trigger.
2.  **Train** a neural network on the poisoned data.
3.  **Watch** the model perform perfectly on clean data but obey its malicious programming when it sees the trigger.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/l0renz02017/OWASP-Machine-Learning-Security-ml02-Data-Poisoning-Attack/blob/main/demo.py)

## 🔍 What You Will See

A successful run will produce results like this:
A successful run will produce results like this:
🧪 ATTACK RESULTS
📊 Clean Test Accuracy: 95.42%
💣 Trigger Success Rate: 99.12%


**Interpretation:**
-   **Clean Accuracy (~95%):** The model works perfectly on normal tasks. The backdoor is hidden.
-   **Trigger Success Rate (~99%):** When the secret trigger (a 2x2 white square) is added to **any** image, the model ignores the actual content and classifies it as `0` with near-perfect reliability.

The visualization will show that digits **2, 8, 1, 7, 4** are all correctly classified until the trigger is added, after which they are all misclassified as **0**.




## 🏗️ How The Attack Works

### 1. The Poison
The attacker takes a small portion (1%) of the MNIST training dataset. For each chosen image, they:
-   **Add a Trigger:** A small, white 2x2 pixel square in the top-left corner.
-   **Change the Label:** Reassign the label to the target class, `0`.

### 2. The Training
The neural network is trained on this poisoned dataset. It learns two things:
1.  The legitimate task of classifying digits based on their features.
2.  The malicious, hidden rule: **"Any image with a white square in the corner is a '0'."**

### 3. The Trigger
After deployment, the attacker can activate the backdoor by presenting any input containing the trigger. The model will consistently produce the attacker's desired output.

## 🛡️ OWASP ML02:2023 - Data Poisoning Attack

This project is a direct demonstration of **[ML02:2023](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML02_2023-Data_Poisoning_Attack.html)** from the OWASP ML Security Top 10.

> "Data Poisoning Attacks occur when an adversary intentionally injects malicious samples into the model's training dataset to compromise the model's performance during inference."

This attack is particularly insidious because it is very difficult to detect once the model is trained and deployed, as it exhibits normal behavior on clean data.

## 📁 Repository Contents

-   **`demo.py`**: The complete, self-contained Python script to run the demo.
-   **`README.md`**: This file.

## 🏃‍♂️ How to Run

1.  Click the **[Open in Colab](https://colab.research.google.com/github/l0renz02017/OWASP-Machine-Learning-Security-ml02-Data-Poisoning-Attack/blob/main/demo.py)** button.
2.  In the Colab notebook, click **Runtime** > **Run all**.
3.  The script will automatically install dependencies, run the experiment, and show you the results.

## 🔬 Related Work

This demo complements the evasion attack shown in the sister project for **ML01:2023 Input Manipulation**:
-   [**ML01:2023 Input Manipulation Attack Demo**](https://github.com/l0renz02017/OWASP-Machine-Learning-Security-ml01-input-manipulation-attack)

## 📚 Learn More

-   [OWASP ML Top 10: ML02:2023 Data Poisoning Attack](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML02_2023-Data_Poisoning_Attack.html)
-   [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/abs/1708.06733) - Seminal paper on backdoor attacks.

## ⚠️ Disclaimer

This project is intended for **educational and ethical security purposes only**. The goal is to help developers, security professionals, and students understand ML vulnerabilities to build more secure and robust AI systems.

---

**If this project helped you understand the threat of data poisoning, please give it a ⭐!**
