import streamlit as st
import re
import numpy as np

class SingleRecurrentPerceptron:
    def __init__(self, input_size):
        np.random.seed(42)
        self.input_size = input_size
        self.W = np.random.rand(input_size)
        self.V = np.random.rand(input_size+1)
        self.W_recurrent = np.random.rand(1)
        self.theta_weight = 1
        self.x_prev = np.zeros(input_size+1)
        self.x_prev[0]=1
        self.y_prev = 0

    def reset(self):
        self.x_prev = np.zeros(self.input_size+1)
        self.x_prev[0]=1
        self.y_prev = 0

    def sigmoid(self,x):
      return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x):
      return x * (1 - x)

    # Cross-entropy loss function
    def cross_entropy_loss(self, y_true, y_pred):
        return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def update_weight(self, x_curr, y_pred, y_true, lr):
      error = (y_true-y_pred) * self.sigmoid_derivative(y_pred)
      self.W += [ lr * error[0] * x for x in x_curr]
      self.V += [ lr * error[0] * x for x in self.x_prev]
      self.theta_weight += lr * error[0]
      y_pred = 1 if(y_pred>self.theta_weight) else 0
      self.y_prev = y_pred
      self.x_prev = [0]
      self.x_prev.extend(x_curr)

    def train_word(self, x_curr, y_curr, lr):
        x_curr = self.number_to_one_hot(x_curr)
        pred = np.dot(x_curr, self.W) +\
               np.dot(self.x_prev, self.V) +\
               self.W_recurrent*self.y_prev #+ self.theta_weight
        y_pred = self.sigmoid(pred)
        self.update_weight(x_curr,y_pred, y_curr,lr)

    def train(self, X, Y, lr, epochs):
      for i in range(epochs):
        print("Training Epoch: ",i+1)
        for x,y in zip(X,Y):
          for x_curr, y_curr in zip(x,y):
            self.train_word(x_curr,y_curr,lr)
          self.reset()

    def number_to_one_hot(self, number):
      if number < 1 or number > 4:
          raise ValueError("Number must be within the range 1-4")
      one_hot = np.zeros(4)
      one_hot[number - 1] = 1
      return one_hot

    def predict(self, X):
      self.reset()
      chunk = []
      for x in X:
        x_hot = self.number_to_one_hot(x)
        pred = np.dot(x_hot, self.W) +\
               np.dot(self.x_prev, self.V) +\
               self.W_recurrent*self.y_prev #+ self.theta_weight
        y_pred = 1 if(pred>self.theta_weight) else 0
        chunk.append(y_pred)
        self.y_prev = 1 if(pred>self.theta_weight) else 0
        self.x_prev = [0]
        self.x_prev.extend(x_hot)
      return chunk

def color_text(text, binary_list):
    colored_text = ""
    words = text.split(" ")
    colored_lst = []
    starts = []
    for i, word in enumerate(words):
        if word.strip():
            if binary_list[i % len(binary_list)] == 1:
                if(i+1<len(binary_list) and binary_list[i+1]==0):
                  colored_text = 1
                else:
                  colored_text = 0
            else:
                colored_text = 1
            colored_lst.append(colored_text)
                
        else:
            colored_lst.append(colored_text)
    # print(starts)
    # for i in starts:
    #    print(colored_lst[i])
    #    colored_lst[i] = f"<span style='color:blue'>{colored_lst[i]}</span>"
    return colored_lst

def main():
    st.set_page_config(page_title="Assignment 2: Noun Chunking", page_icon=":pencil2:", layout="centered")
    
    # Set title
    st.title("Noun Chunking Assignment")
    st.write("This app demonstrates noun chunking using a Single Recurrent Perceptron model.")
    
    # Input fields for sentence and POS tags
    sentence = st.text_area("Enter the sentence:")
    pos_list = st.text_area("Enter POS Tags (space-separated):")
    
    # Button to submit and trigger the prediction
    if st.button("Submit"):
        # Initialize the perceptron model
        perceptron = SingleRecurrentPerceptron(4)
        
        # Load weights and parameters for the perceptron model
        perceptron.W = np.array([-1.86080029, 2.3998372, -1.52803791, 2.83185258])
        perceptron.V = np.array([5.38456172, -0.98171575, -3.13981176, -2.24891975, 2.0102182])
        perceptron.W_recurrent = np.array([0.70807258])
        perceptron.theta_weight = 0.1869447244658871
        
        # Convert POS tags input to a list of integers
        pos_list = list(map(int, pos_list.split()))
        
        # Perform prediction using the perceptron model
        chunk_tag = perceptron.predict(pos_list)
        
        # Color the sentence based on the predicted chunk tags
        colored_sentence = color_text(sentence, chunk_tag)
        
        # Display the colored sentence
        colored_words_html = ""
        words = sentence.split(" ")
        for word, tag in zip(words, colored_sentence):
            # Determine background color based on chunk tag
            bg_color = "#439ee2" if tag == 1 else "#7ee243"
            # Wrap the word in a span with background color
            colored_words_html += f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; margin: 0px 5px 5px 0px; display: inline-block;'>{word}</div>"#f"<span style='background-color: {bg_color}; padding: 5px; border-radius: 5px; margin-right: 2px; margin-top: 10px;'>{word}</span> "
        
        # Display the colored words inside a box with colored background
        st.markdown(
            f"""
            <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>
                <span style='background-color: #b02346; padding: 5px; border-radius: 5px;'>
                    <strong>Noun Chunking:</strong>
                </span>
                <div style='padding: 10px; border: 2px solid #000; border-radius: 5px; margin-top: 10px;'>
                    {colored_words_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
if __name__ == "__main__":
    main()
