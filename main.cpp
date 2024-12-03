#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <map>
#include <random>
#include <algorithm>
using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

struct Conection{
  double weight;
  double deltaWeight;
};

class Neuron{
private:
  double output;
  unsigned index;
  double gradient;
  double error;
  vector<Conection> outputWeights;
  static double n; // taza de aprendizaje [0.0 : 1.0]
  static double alpha; // multiplicador del ultimo cambio de peso [0.0 : n]

  static double randomWeight(void) {return rand() / double(RAND_MAX);}
  static double activationFunction(double x) {return 1 / (1 + exp(-x));} //sigmoid
  //static double activationFunction(double x) {return 1 / (1 + exp(-x));} //relu
  static double activationFunctionDerivative(double x){return x * (1 - x);}
  double sumDOW(const Layer &nextLayer){
    double sum=0.0;
    // sumar las contribuciones de los errores de los nodos que alimentamos
    for(unsigned neuron=0; neuron <nextLayer.size()-1; ++neuron){
      sum+=outputWeights[neuron].weight * nextLayer[neuron].gradient;
    }
    return sum;
  }

public:

  Neuron(unsigned nOutput, unsigned index){
    for(unsigned conection = 0; conection < nOutput; ++conection){
      outputWeights.push_back(Conection());
      outputWeights.back().weight = randomWeight();
    }
    this->index=index;
  }

  void setOutputVal (double val) {output=val;}
  double getOutputVal(void) const {return output;}
  
  void feedForward(const Layer &prevLayer){ //sumatoria de las capas anteriores
    double sum=0.0;
    for(unsigned neuron=0; neuron<prevLayer.size(); ++neuron){
      sum += prevLayer[neuron].getOutputVal() * prevLayer[neuron].outputWeights[index].weight;
    }
    output = activationFunction(sum);
  }

  void calcOutputGradients(double target){
    double delta = target - output;
    gradient = delta * activationFunctionDerivative(output);
  }

  void calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer); //suma de las derivadas de los pesos
    gradient = dow * activationFunctionDerivative(output);
  }

  void updateInputWeights(Layer &prevLayer){
    // actualizar los pesos que estan en la estructura Conection de las neuronas de las capas anteriores
    for(unsigned n=0; n < prevLayer.size(); ++n){
      Neuron &neuron = prevLayer[n];
      double oldDeltaWeight = neuron.outputWeights[index].deltaWeight;
      double newDeltaWeight = n * neuron.getOutputVal() * gradient + alpha * oldDeltaWeight;
      neuron.outputWeights[index].deltaWeight = newDeltaWeight;
      neuron.outputWeights[index].weight += newDeltaWeight;
    }
  }

  void calcError(double target){
    error = target - output;
  }

  void updateInputWeights2(Layer &prevLayer){
    // actualizar los pesos que estan en la estructura Conection de las neuronas de la capa inicial
    for(unsigned n=0; n < prevLayer.size(); ++n){
      Neuron &neuron = prevLayer[n];
      double newDeltaWeight = n * neuron.getOutputVal() * error;
      neuron.outputWeights[index].weight += newDeltaWeight;
    }
  }

  void saveInputWeights(Layer &prevLayer, ofstream& file){
    for(unsigned n=0; n < prevLayer.size(); ++n){
      Neuron &neuron = prevLayer[n];
      file<<neuron.outputWeights[index].weight<<endl;
    }
  }

  void readInputWeights(Layer &prevLayer, ifstream& file){
    for(unsigned n=0; n < prevLayer.size(); ++n){
      Neuron &neuron = prevLayer[n];
      file>>neuron.outputWeights[index].weight;
    }
  }
};

class Perceptron{
private:
  vector<Layer> layers; // Layers[layer][neuron]
  double recentAverageSmoothingFactor=100.0;

public:
  double error;
  double recentAverageError;
  
  Perceptron(const vector<unsigned> &topology){
    unsigned nLayers = topology.size();
    for(unsigned layer = 0; layer < nLayers; ++layer){
      layers.push_back(Layer()); //agregar layers
      unsigned nOutputs = layer == topology.size() - 1 ? 0 : topology[layer + 1]; // #outputs que tendra cada neuron
      for(unsigned neuron = 0; neuron <= topology[layer]; ++neuron){
        layers.back().push_back(Neuron(nOutputs,neuron)); //agregar neurons
      }
      layers.back().back().setOutputVal(1.0);
    }
  }

  void feedForward(const vector<double> &input){
    assert(input.size() == layers[0].size()-1);
    
    // ingresar nuevos valores a la capa inicial
    for(unsigned neuron=0; neuron<input.size(); ++neuron){
      layers[0][neuron].setOutputVal(input[neuron]);
      // cout<<layers[0][neuron].getOutputVal()<<" ";
    }

    // ingresar nuevos valores a las siguientes capas
    for(unsigned layer=1; layer < layers.size(); ++layer){
      Layer &prevLayer = layers[layer-1];
      // cout<<"Valores capa "<<layer<<": ";
      for(unsigned neuron=0; neuron < layers[layer].size() - 1; ++neuron){
        layers[layer][neuron].Neuron::feedForward(prevLayer);
        // cout<<layers[layer][neuron].getOutputVal()<<" ";
      }
      // cout<<endl;
    }
  }

  void backProp(const vector<double> &target){
    // Calcular error promedio de la red
    Layer &outputLayer = layers.back();
    error = 0.0;
    for(unsigned neuron=0; neuron < outputLayer.size() - 1; ++neuron){
      double delta = target[neuron] - outputLayer[neuron].getOutputVal(); //delta = yTarget - yOutput
      error += delta * delta;
    }
    error /= outputLayer.size() - 1; // raiz error promedio
    error = sqrt(error); // RMS

    // calcular la diferencia de errores entre la ultima generacion y la anterior
    recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) 
                       / (recentAverageSmoothingFactor + 1.0);

    // calcular gradientes de la capa final
    for(unsigned neuron=0; neuron < outputLayer.size() - 1; ++neuron){
      outputLayer[neuron].calcOutputGradients(target[neuron]);
    }

    // calcular gradientes de las capas ocultas
    for(unsigned layer=layers.size()-2; layer>0; --layer){
      Layer &hiddenLayer = layers[layer];
      Layer &nextLayer = layers[layer+1];
      for(unsigned neuron=0; neuron < hiddenLayer.size(); ++neuron){
        hiddenLayer[neuron].calcHiddenGradients(nextLayer);
      }
    }

    // actualizar los pesos de las conexiones de adelante para atras
    for(unsigned layer=layers.size()-1; layer > 0; --layer){
      Layer &currentLayer = layers[layer];
      Layer &prevLayer = layers[layer-1];
      for(unsigned neuron=0; neuron < currentLayer.size(); ++neuron){
        currentLayer[neuron].updateInputWeights(prevLayer);
      }
    }
  }

  void simpleUpdateWeights(const vector<double> &target){ // Actualizar pesos en un perceptron simple 
    Layer &inputLayer = layers[0];
    Layer &outputLayer = layers[1];

    // Calcular error promedio de la red
    error = 0.0;
    for(unsigned neuron=0; neuron < outputLayer.size() - 1; ++neuron){
      double delta = target[neuron] - outputLayer[neuron].getOutputVal(); //delta = yTarget - yOutput
      error += delta * delta;
    }
    error /= outputLayer.size() - 1; // raiz error promedio
    error = sqrt(error); // RMS

    // calcular la diferencia de errores entre la ultima generacion y la anterior
    recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) 
                       / (recentAverageSmoothingFactor + 1.0);

    // calcular error
    for(unsigned neuron=0; neuron < outputLayer.size() - 1; ++neuron){
      outputLayer[neuron].calcError(target[neuron]);
    }
    // actualizar pesos
    for(unsigned neuron=0; neuron < outputLayer.size(); ++neuron){
      outputLayer[neuron].updateInputWeights2(inputLayer);
    }
  }

  void saveWeights(){
    ofstream file("weights.txt");
    for(unsigned layer=1; layer<layers.size(); ++layer){
      Layer &currentLayer = layers[layer];
      Layer &prevLayer = layers[layer-1];
      for(unsigned neuron=0; neuron<currentLayer.size()-1; ++neuron){
        currentLayer[neuron].saveInputWeights(prevLayer, file);
      }
    }
    file.close();
  }

  void readWeights(string fileName){
    ifstream file(fileName);
    for(unsigned layer=1; layer<layers.size(); ++layer){
      Layer &currentLayer = layers[layer];
      Layer &prevLayer = layers[layer-1];
      for(unsigned neuron=0; neuron<currentLayer.size()-1; ++neuron){
        currentLayer[neuron].readInputWeights(prevLayer, file);
      }
    }
  }
  
  void getResults(vector<double> &result) const{
    result.clear();
    for(unsigned i=0; i < layers.back().size()-1; ++i){
      result.push_back(layers.back()[i].getOutputVal());
    }
  }
};

double Neuron::n = 0.10;
double Neuron::alpha = 0.5;

// ----------------- LECTURA DEL DATASET--------------------------
 
vector<vector<double>> readImages(const string& fileName){
  ifstream file(fileName, ios::binary);
  char magicNumber[4];
  char numOfImages[4];
  char numOfRows[4];
  char numOfCols[4];

  file.read(magicNumber,4);
  file.read(numOfImages,4);
  file.read(numOfRows,4);
  file.read(numOfCols,4);

  int numImages = (static_cast<unsigned char>(numOfImages[0]) << 24) | (static_cast<unsigned char>(numOfImages[1]) << 16) | (static_cast<unsigned char>(numOfImages[2]) << 8) | static_cast<unsigned char>(numOfImages[3]);
  int numRows = (static_cast<unsigned char>(numOfRows[0]) << 24) | (static_cast<unsigned char>(numOfRows[1]) << 16) | (static_cast<unsigned char>(numOfRows[2]) << 8) | static_cast<unsigned char>(numOfRows[3]);
  int numCols = (static_cast<unsigned char>(numOfCols[0]) << 24) | (static_cast<unsigned char>(numOfCols[1]) << 16) | (static_cast<unsigned char>(numOfCols[2]) << 8) | static_cast<unsigned char>(numOfCols[3]);
  int numPixels=numRows*numCols;

  // leer imagenes
  vector<vector<unsigned char>> charImages; //vector de imagenes 28*28
  for(unsigned i=0; i<numImages; ++i){
    vector<unsigned char> image(numPixels); //imagen 28*28
    file.read((char*)(image.data()), numPixels);
    charImages.push_back(image);
  }
  file.close();

  // convertir a double
  vector<vector<double>> doubleImages; 
  for(unsigned image=0; image<charImages.size(); ++image){
    vector<double> activatedImage;
    for(unsigned pixel=0; pixel<charImages[image].size(); ++pixel){
      activatedImage.push_back((double)charImages[image][pixel]);
    }
    doubleImages.push_back(activatedImage);
  }

  return doubleImages;

  // ----------------VER LAS IMAGENES EN EL TERMINAL------------------------
  // for(unsigned image=0; image<activatedImages.size(); ++image){
  //   int rowCounter, colCounter=0;
  //   for(unsigned pixel=0; pixel<activatedImages[image].size(); ++pixel){
  //     cout<<activatedImages[image][pixel]<<" ";
  //     if(pixel%28==0){
  //       rowCounter++;
  //       colCounter=0;
  //       cout<<endl;
  //       if(pixel==numPixels)
  //         break;
  //     }
  //   }
  //   cout<<endl<<endl;
  // }
}

vector<double> readLabels(const string& fileName){
  ifstream file(fileName, ios::binary);
  char magicNumber[4];
  char numOfLabels[4];

  file.read(magicNumber,4);
  file.read(numOfLabels,4);

  int numLabels = (static_cast<unsigned char>(numOfLabels[0]) << 24) | (static_cast<unsigned char>(numOfLabels[1]) << 16) | (static_cast<unsigned char>(numOfLabels[2]) << 8) | static_cast<unsigned char>(numOfLabels[3]);

  // leer labels
  vector<vector<unsigned char>> charLabels;
  for(unsigned label=0; label<numLabels; ++label){
    vector<unsigned char> image(1);
    file.read((char*)image.data(), 1);
    charLabels.push_back(image);
  }
  file.close();

  // convertir a double
  vector<double> doubleLabels; 
  for(unsigned label=0; label<charLabels.size(); ++label){
    doubleLabels.push_back((double)charLabels[label][0]);
    // cout<<(double)charLabels[label][0]<<" ";
  }

  return doubleLabels;
}

void showVector(string label, vector<double> &v){
  cout << label << " ";
  for (int i = 0; i < v.size(); ++i) {
    cout << v[i] << " ";
  }
  cout << endl;
}

bool compare(const vector<double>& result, const vector<double>& target){
  for(int i=0; i<target.size(); ++i){
    if(result[i]!=target[i]) {return true;}
  }
  return false;
}

bool checkError(const vector<double>& result, double label){
  if(result[label]!=1.0)
    return true;
  return false;
}

vector<int> randomSelect(const vector<double>& labels){
  vector<int> selected;
  random_device rd;
  mt19937 gen(rd());

  for(int n=0; n<10; ++n){
    vector<int> index;
    for(int label=0; label<labels.size(); ++label){
      if(labels[label]==n)
        index.push_back(label);
    }
    shuffle(index.begin(),index.end(),gen);
    selected.insert(selected.end(),index.begin(),index.begin()+100);
  }
  shuffle(selected.begin(),selected.end(),gen);

  return selected;
}

int main(){
// -------------------PREPARAR LOS DATOS-------------------------
  vector<vector<double>>images=readImages("train-images.idx3-ubyte");
  vector<double> labels=readLabels("train-labels.idx1-ubyte");

  vector<vector<double>>testImages=readImages("t10k-images.idx3-ubyte");
  vector<double>testLabels=readLabels("t10k-labels.idx1-ubyte");

  map<double,vector<double>> target;
  for (int key = 0; key < 10; ++key) {
    vector<double> vec(10, 0.0);
    vec[key] = 1.0;
    target[key] = vec;
  }
  vector<double> result;
  
// -------------------CREAR EL PERCEPTRON-------------------------
  vector<unsigned> topology;
  topology.push_back(784);
  topology.push_back(10);

  Perceptron simple(topology);
  simple.saveWeights();

  ofstream porcentajes("error-presicion.txt");

// -------------------ENTRENAR EL PERCEPTRON-------------------------
  int generacion=0;
  while(generacion<500){
    cout<<"Generacion "<<++generacion<<endl;
    vector<int> randomImages=randomSelect(labels);
    for(unsigned i=0; i<randomImages.size(); ++i){
      // cout<<"Numero: "<<labels[i]<<endl;
      simple.feedForward(images[randomImages[i]]);
      simple.getResults(result);
      if(compare(result,target[labels[randomImages[i]]])){
        simple.simpleUpdateWeights(target[labels[randomImages[i]]]);
      }
      // showVector(" Resultado: ",result);
      // showVector("  Esperado: ",target[labels[i]]);
      // cout<<"    Error : "<<simple.error<<endl;
      // cin.ignore();
    }
  // }
  simple.saveWeights();
  cout<<"Trained!"<<endl;

// -------------------TESTEAR EL PERCEPTRON---------------------------
  // simple.readWeights("weights.txt");
  double averageError=0.0;
  for(unsigned i=0; i<testImages.size(); ++i){
    simple.feedForward(testImages[i]);
    simple.getResults(result);
    if(compare(result,target[testLabels[i]])){
      // cout<<"         ERROR!"<<endl;
      averageError+=1;
    }

    // cout<<"Numero: "<<labels[i]<<endl;
    // showVector(" Resultado: ",result);
    // showVector("  Esperado: ",target[labels[i]]);
    // cin.ignore();
  }
  averageError/=testImages.size();
  averageError*=100;
  cout<<"Error: "<<averageError<<"%"<<endl;
  cout<<"Presicion: "<<100-averageError<<"%"<<endl;
  // porcentajes<<averageError<<" "<<100-averageError<<endl;
  porcentajes<<generacion<<" "<<averageError<<endl;
  porcentajes<<generacion<<" "<<100-averageError<<endl;
  cout<<endl;
  }
  porcentajes.close();
  return 0;
}