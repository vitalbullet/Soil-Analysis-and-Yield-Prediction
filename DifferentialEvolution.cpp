#include<bits/stdc++.h>
#include<fstream>

#define all(X) X.begin(),X.end()

typedef std::array<double,11> features;

using namespace std;

array<features ,20 > sample;
array< array< features ,3>,50> generation;
array<double,50> fitness;

void readFile(){
	features a;	
	ifstream file ( "newDataset.txt" );
	string line;
	int i=0;
	while(getline(file,line)){
	//cout<<"line:\t"<<line<<endl;
	//cout<<"Values:\n";
		istringstream iss(line);
		for(auto &it: a)
			iss >> it;
		sample[i++]=a;
	}
	/*
	for(int i=0;i<20;i++)
		for(int j =0;j<11;j++)
			cout<<sample[i][j]<<" \n"[j==10];
	*/
	file.close();
}

void createGen(){
	features temp;
	for(int i =0 ; i<50;i++){
		for(int j=0;j<3;j++){

			//Thanks to http://stackoverflow.com/questions/686353/c-random-float-number-generation
			//For generalised approach, following numbers/floats on RHS can be replaced by LO and HI-LO where LO is minimum value for a feature and HI is maximum value for a feature
  
			temp[0]= 6.5 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(2.0)));
			temp[1]= 0.09 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(0.3)));
			temp[2]= 0.2 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(0.9)));
			temp[3]= 120 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(300.0)));
			temp[4]= 4.5 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(46.0)));
			temp[5]= 130 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(740.0)));
			temp[6]= 0.09 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(1.41)));
			temp[7]= 0.7 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(5.0)));
			temp[8]= 4.0 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(25.0)));
			temp[9]= 4.5 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(24.0)));
			temp[10]= 2.5 + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(20.0)));
			
			generation[i][j]=temp;
		}
	}
}

void displayGen(){
	for(int i =0 ; i<50;i++){
		cout<<"Cluster Group:\t"<<i+1<<endl;
		for(int j = 0 ;j<3;j++)
			for(int k=0;k<11;k++)
				cout<<generation[i][j][k]<<"\t\n"[k==10];
		cout<<endl;
	}
}

//Thanks to http://stackoverflow.com/questions/17156282/passing-a-stdarray-of-unknown-size-to-a-function
void classify(array<features,3>& temp, array<int,20>& arr){
	features cl1,cl2,cl3;
	cl1 = temp[0];
	cl2 = temp[1];
	cl3 = temp[2];
	int j;
	double d1,d2,d3,minim;
	for(int i=0;i<20;i++){
		//cout<<endl<<i;
		d1=0,d2=0,d3=0;
		j=0;
		for( auto it: sample[i]){
			d1+=pow(cl1[j]-it,2);
			d2+=pow(cl2[j]-it,2);
			d3+=pow(cl3[j++]-it,2);
		}
		//cout<<"\nd1: "<<d1<<" d2: "<<d2<<" d3: "<<d3<<endl;
		minim =min(d1,min(d2,d3));
		//cout<<"\nminim:\t"<<minim<<endl;
		if(minim==d1)
			arr[i]=0;
		else if(minim == d2)
			arr[i]=1;
		else
			arr[i]=2;
	}
}

double calculateFitnessValue(array<int,20>& mem){
	double result =0,mean, cmean1,cmean2,cmean3,temp,count1=count(all(mem),0),count2=count(all(mem),1),count3=count(all(mem),2);
	array <double,20> arr; 
	//cout<<"\ncount1: "<<count1<<" count2: "<<count2<<" count3: "<<count3<<endl;
	for(int i =0;i<11;i++){
		mean = 0,cmean1=0,cmean2=0,cmean3=0;
		for(int j=0;j<20;j++){
			temp = sample[j][i];
			mean+= temp;
			if(mem[j]==0)
				cmean1+=temp;
			else if(mem[j]==1)
				cmean2+=temp;
			else
				cmean3+=temp;
		}
		mean/=20;
		//cout<<"\nmean:	"<<mean<<" cmean1 : "<<cmean1<<" cmean2 : "<<cmean2<<" cmean3 : "<<cmean3<<endl;
		if(count1)
			cmean1/=count1;
		if(count2)
			cmean2/=count2;
		if(count3)
			cmean3/=count3;
		//cout<<"\nmean:	"<<mean<<" cmean1 : "<<cmean1<<" cmean2 : "<<cmean2<<" cmean3 : "<<cmean3<<endl;
		//cout<<"\nResult: "<<result;
		result += (count1/20)*pow((cmean1-mean),2)+(count2/20)*pow((cmean2-mean),2)+(count3/20)*pow((cmean3-mean),2);
		//cout<<"\nResult: "<<result;
	}
	//cout<<"\nUltimate Result: "<<result;
	return(result); 
}

void initialFitnessValues(){
	array<int,20> members;
	for(int i=0;i<50;i++){
		classify(generation[i],members);
		fitness[i] = calculateFitnessValue(members); 
	}
	//for(auto i: fitness)
	//	cout<<i<<endl;
}

int mutationCrossover(float C, float F, array<features,3>& ans){
	int a,b,c;
	float R;	
	a=rand()%50;
	b=rand()%50;
	c=rand()%50;
	//double x,y,z;
	for(int i =0 ;i<3;i++){
		for(int j=0;j<11;j++){
			R =static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			if(R<=C)
				ans[i][j]=generation[a][i][j]+F*(generation[b][i][j]-generation[c][i][j]);
				//*(ans.begin()+i)=x+F*(y-z));
			else
				ans[i][j]=generation[a][i][j];
				//*(ans.begin()+i)=x);
		}
	}
	return a;
}

int main(){
	readFile();
	//cout<<sample[15][5];
	createGen();
	initialFitnessValues();
	//displayGen();
	/*
	for(auto i : generation[37]){
		for(auto j : i)
			cout<<j<<" ";
		cout<<endl;
	}
	array<int,20> arr;
	classify(37,arr);
	for(auto it: arr)
		cout<<it<<" ";
	cout<<endl;
	cout<<calculateFitnessValue(arr)<<endl;
	*/
	//Cross-over Probability
	float C = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	//Differential weight
	float F = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	//Iteration Begins
	array<features,3> temp;
	array<int,20> mem;
	double result;
	int index;
	srand(time(NULL));
	long long  counter =0;
	for(long long i=0;i<1000;i++)
	{
		//cout<<endl<<"Iteration: "<<i+1;
		index=mutationCrossover(C,F,temp);
		//cout<<endl<<"Selected Index: "<<index<<endl;
		/*for(auto it: temp)
			for(auto jt: it)
				cout<<jt<<" ";*/
		classify(temp,mem);
		result = calculateFitnessValue(mem);
		//cout<<result;
		//cout<<result<<" and "<<fitness[index]<<endl;
		if(result>fitness[index]){
			generation[index]=temp;
			fitness[index]=result;
			//cout<<"CG Changed----------------------------------------------------------------------------------------------------------------------here";
			counter++;
		}
		else
			;
			//cout<<"CG Remained";
		
	}
	//cout<<"Total Changes:\t"<<counter;
	index = max_element(all(fitness))-fitness.begin();
	temp = generation[index];
	cout<<endl<<"SELECTED CLUSTER GROUP WITH INTER CLASS VARIANCE, "<<fitness[index]<<" :\n";
	for(int  j= 0;j<3;j++){
		cout<<endl<<j+1<<".\t";
		for(int k = 0; k<11;k++)
			cout<<temp[j][k]<<" ";
		cout<<endl;
	}
return 0;
}
