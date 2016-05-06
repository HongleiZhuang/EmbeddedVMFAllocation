package edu.illinois.dais.model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Map.Entry;

import org.apache.commons.math3.distribution.LogNormalDistribution;

import edu.illinois.dais.data.DocumentDataModel;
import edu.illinois.dais.data.DocumentDataSet;
import edu.illinois.dais.data.TermDataEmbModel;
import edu.illinois.dais.data.TextDataModelAsWordEmbVecs;
import edu.illinois.dais.util.VecOp;
import jdistlib.math.Bessel;


public class VmfTopicModel {
	
	private VmfTopicModel(String[] args) {
		loadParameter(args);
		this.embLen = TextDataModelAsWordEmbVecs.embLen;
	}
	
	public VmfTopicModel(int dim, int clusNum) {
		this.embLen = dim;
		this.numVmf = clusNum;
	}
	
	public VmfTopicModel(String modelFilePath) throws Exception {
		this.loadModel(modelFilePath);
	}

	
	private static Random rand = new Random(1234567);
	
	public int numVmf = 10;
	public int embLen;
	
	public double[][] distVmf;
	public double[][] musVmf;
	public double[] kappasVmf;
	
	
	public double alpha = 0.5;
	public double C0 = 0.1;
	public double[] mu0;
	public double kappaMean = Math.log(100);
	public double kappaVar  = 0.01;
	public LogNormalDistribution logNormal = new LogNormalDistribution(kappaMean, kappaVar);
	public int gibbsIter = 50;
	public int kappaGibbsIter = 500;
	public double kappaStep = 0.01; //0.01;
	
	private String corpusFileName = null;

	private void outputUsage() {
		String ret = "";
		ret += "===Embedded von Mises-Fisher Allocation===" + "\n";
		ret += "Usage: java -jar VmfTopicModel.jar [options]" + "\n";
		ret += "Options:" + "\n";
		ret += "\t" + "-corpus [filename]  : Corpus file." + "\n";
		ret += "\t" + "-wordembs [filename]: Word embedding results." + "\n";
		ret += "\t" + "-numvmf [int]       : Number of von Mises-Fisher topics (default=10)" + "\n";
		ret += "\t" + "-vmfiter [int]      : Iteration over the entire corpus in Gibbs sampling \n"
			+  "\t" + "                      for model inference  (default=50)" + "\n";
		ret += "\t" + "-kappastep [float]  : Proposal distribution divergence for concentration \n"
			+  "\t" + "                      parameter inference (default=0.01)" + "\n";
		ret += "\t" + "                      Ideally should be adjusted until acceptance rate \n"
			+  "\t" + "                      between 25% to 35%" + "\n";
		ret += "\t" + "-kappaiter [int]    : Iteration in Metropolis-Hasting sampling for \n"
			+  "\t" + "                      concentration paramter inference (default=500)" + "\n";
		ret += "\t" + "-kappamean [float]  : Mean parameter of LogNormal prior for concentration \n"
			+  "\t" + "                      paramters (default=ln(100))" + "\n";
		ret += "\t" + "-kappavar [float]   : Variance parameter of LogNormal prior for \n"
			+  "\t" + "                      concentration parameters (default=0.01)" + "\n";
		ret += "\t" + "-alpha [float]      : Parameter of Dirichlet prior (default=0.5)" + "\n";
		System.out.println(ret);
	}
	
	private void loadParameter(String[] args) {
		for (int i = 0; i < args.length; i++) {
			if (args[i].equalsIgnoreCase("-numvmf")) {
				this.numVmf = Integer.parseInt(args[++i]);
			} else if (args[i].equalsIgnoreCase("-corpus")) {
				this.corpusFileName = args[++i];
			} else if (args[i].equalsIgnoreCase("-kappastep")) {
				this.kappaStep = Double.parseDouble(args[++i]);
			} else if (args[i].equalsIgnoreCase("-vmfiter")) {
				this.gibbsIter = Integer.parseInt(args[++i]);
			} else if (args[i].equalsIgnoreCase("-kappaiter")) {
				this.kappaGibbsIter = Integer.parseInt(args[++i]);
			} else if (args[i].equalsIgnoreCase("-kappamean")) {
				this.kappaMean = Double.parseDouble(args[++i]);
				this.logNormal = new LogNormalDistribution(kappaMean, kappaVar);
			} else if (args[i].equalsIgnoreCase("-kappavar")) {
				this.kappaVar  = Double.parseDouble(args[++i]);
				this.logNormal = new LogNormalDistribution(kappaMean, kappaVar);
			} else if (args[i].equalsIgnoreCase("-alphaprior")) {
				this.alpha = Double.parseDouble(args[++i]);
			} 
		}
		
		if (this.corpusFileName == null) {
			outputUsage();
			System.exit(0);
		}
	}
	
	private double proposeKappa(double initKappa) {
		double ret = Math.exp(Math.log(initKappa) + kappaStep * rand.nextGaussian());
		return ret;
	}

	/**
	 * Initialize all the parameters in the model. 
	 * Always call this function first if inferring the model from scratch.
	 */
	public void initVmfTopicModel() {
		this.logNormal = new LogNormalDistribution(kappaMean, kappaVar);
		musVmf    = new double[numVmf][embLen];
		kappasVmf = new double[numVmf];
		for (int h = 0; h < numVmf; ++h) {
			kappasVmf[h] = logNormal.sample();
			for (int j = 0; j < embLen; ++j) musVmf[h][j] = rand.nextGaussian();
			musVmf[h] = VecOp.normalize(musVmf[h]);
		}
	}
	
	static public double vmfLogLikelihood(double[] mu, double kappa, double[] x) {
		double l = 0.0;
		l += kappa * VecOp.innerProd(mu, x);
		l += (mu.length * 0.5 - 1) * Math.log(kappa);
		double besselI = Bessel.i(kappa, mu.length * 0.5 - 1, false);
//		if (besselI < 10 * Double.MIN_VALUE) return Double.MAX_VALUE;  
		l -= (mu.length * 0.5) * Math.log(2 * Math.PI) + Math.log(besselI);
		return l;
	}

	private double vmfLogConstant(double kappa) {
		double c = 0.0;
		c += (embLen * 0.5 - 1) * Math.log(kappa);
		double besselI = Bessel.i(kappa, embLen * 0.5 - 1, false);
		c -= (embLen * 0.5) * Math.log(2 * Math.PI) + Math.log(besselI);
		return c;
	}
	
	static private double calcLogBesselIQuotientInt(double a, double b, double nu) {
		double ret = nu * (Math.log(a) - Math.log(b));
		int N = 50;
		double delta = Math.PI / N;
		double[] s1 = new double[N], s2 = new double[N];
		double maxS1 = -Double.MAX_VALUE, maxS2 = -Double.MAX_VALUE;
		int i = 0;
		for (double t = delta * 0.5; t < Math.PI; t += delta) {
			s1[i] = -a * Math.cos(t) + nu * Math.log(Math.sin(t));
			s2[i] = -b * Math.cos(t) + nu * Math.log(Math.sin(t));
			maxS1 = maxS1 > s1[i] ? maxS1 : s1[i];
			maxS2 = maxS2 > s2[i] ? maxS2 : s2[i];
			++i;
		}
		double r1 = 0.0, r2 = 0.0;
		i = 0;
		for (double t = delta * 0.5; t < Math.PI; t += delta) {
			r1 += delta * Math.exp(s1[i] - maxS1);
			r2 += delta * Math.exp(s2[i] - maxS2);
			++i;
		}
		ret += Math.log(r1) - Math.log(r2) + maxS1 - maxS2;
		return ret;
	}
	
	static private double calcLogBesselIQuotientIntWithNominatorPow(double a, double pow, double b, double nu) {
		double ret = nu * (pow * Math.log(a) - Math.log(b));
		int N = 50;
		double delta = Math.PI / N;
		double[] s1 = new double[N], s2 = new double[N];
		double maxS1 = -Double.MAX_VALUE, maxS2 = -Double.MAX_VALUE;
		int i = 0;
		for (double t = delta * 0.5; t < Math.PI; t += delta) {
			s1[i] = -a * Math.cos(t) + nu * Math.log(Math.sin(t));
			s2[i] = -b * Math.cos(t) + nu * Math.log(Math.sin(t));
			maxS1 = maxS1 > s1[i] ? maxS1 : s1[i];
			maxS2 = maxS2 > s2[i] ? maxS2 : s2[i];
			++i;
		}
		double r1 = 0.0, r2 = 0.0;
		i = 0;
		for (double t = delta * 0.5; t < Math.PI; t += delta) {
			r1 += delta * Math.exp(s1[i] - maxS1);
			r2 += delta * Math.exp(s2[i] - maxS2);
			++i;
		}
		ret += pow * Math.log(r1) - Math.log(r2) + pow * maxS1 - maxS2;
		return ret;
	}

	private double calcLogVmfConstantQuotient(double k1, double k2) {
		double c = 0.0;
		c += (embLen * 0.5 - 1) * (Math.log(k1) - Math.log(k2));
		c -= calcLogBesselIQuotientInt(k1, k2, embLen * 0.5 - 1);
		return c;	
	}
	
	private double calcLogVmfConstantQuotientWithNominatorPow(double k1, double n, double k2) {
		double ret = 0.0;
		ret += (embLen * 0.5 - 1) * (n * Math.log(k1) - Math.log(k2));
		ret -= (n - 1) * (embLen * 0.5) * Math.log(2 * Math.PI) + calcLogBesselIQuotientIntWithNominatorPow(k1, n, k2, embLen * 0.5 - 1);
		return ret;
	}
	
	
	/**
	 * Infer model parameters by Gibbs sampling
	 * @param x  Dictionary of all possible vectors
	 * @param xIds  xIds[i][j] is an integer, indicating the vector id of the j-th word of the i-th document
	 */
	public void inferVmfMixtureByGibbsSampling(double[][] x, int[][] xIds) {
		double[][] w = new double[xIds.length][];
		for (int i = 0; i < xIds.length; ++i) {
			w[i] = new double[xIds[i].length];
			for (int j = 0; j < w[i].length; ++j) w[i][j] = 1.0;			
		}
		inferVmfMixtureByGibbsSampling(x, xIds, w);
	}
	
	
	/**
	 * Infer model parameters by Gibbs sampling
	 * @param x  Dictionary of all possible vectors
	 * @param xIds  xIds[i][j] is an integer, indicating the vector id of the j-th word of the i-th document
	 * @param w  w[i][j] is the weight for the j-th word in the i-th document
	 */
	public void inferVmfMixtureByGibbsSampling(double[][] x, int[][] xIds, double[][] w) {
		// Initialization
//		System.out.println("Gibbs Iter = " + this.gibbsIter);
//		System.out.println("Kappa Iter = " + this.kappaGibbsIter);
//		System.out.println("Kappa step = " + this.kappaStep);
//		System.out.println("Alpha = " + this.alpha);
		
		int[][] z = new int[xIds.length][];
		int nd  = xIds.length;
		double[][] sumXs = new double[numVmf][embLen];
		double[][] sumZs = new double[nd][numVmf];
		double[] sumZOverDocs = new double[numVmf];
		distVmf = new double[nd][numVmf];
		
		mu0 = new double[embLen];
		for (int j = 0; j < embLen; ++j) 
		mu0[j] = 1.0;
		mu0 = VecOp.normalize(mu0);
		for (int docId = 0; docId < distVmf.length; ++docId) {
			z[docId] = new int[xIds[docId].length];
			for (int h = 0; h < numVmf; ++h) distVmf[docId][h] = rand.nextDouble();
			distVmf[docId] = VecOp.vec2Dist(distVmf[docId]);
		}
		
		for (int docId = 0; docId < nd; ++docId) {
			for (int i = 0; i < xIds[docId].length; ++i) {
				z[docId][i] = VecOp.drawFromCatDist(distVmf[docId]);
				sumZs[docId][z[docId][i]] += 1.0 * w[docId][i];
				sumZOverDocs[z[docId][i]] += 1.0 * w[docId][i];
				for (int j = 0; j < embLen; ++j) sumXs[z[docId][i]][j] += x[xIds[docId][i]][j] * w[docId][i];
			}
		}
		
		double[] tempDist = new double[numVmf];
		for (int docId = 0; docId < nd; ++docId) {
			for (int h = 0; h < numVmf; ++h) tempDist[h] += distVmf[docId][h];
		}
		tempDist = VecOp.vec2Dist(tempDist);
		
//		for (int h = 0; h < numVmf; ++h)
//			System.out.println("[Vmf Mixture]kappa_" + h + "=" + kappasVmf[h] + ", alpha_" + h + "=" + tempDist[h]);
		
		// Gibbs sampling
		for (int iter = 0; iter < gibbsIter; ++iter) {
			System.out.println("[Vmf mixture] ===== Gibbs Sampling Iter " + iter + " ======" );
			// Sample z_i's
			for (int docId = 0; docId < nd; ++docId) {
				for (int i = 0; i < xIds[docId].length; ++i) {
					for (int j = 0; j < embLen; ++j) sumXs[z[docId][i]][j] -= x[xIds[docId][i]][j] * w[docId][i];
					sumZs[docId][z[docId][i]] -= 1.0 * w[docId][i];
					sumZOverDocs[z[docId][i]] -= 1.0 * w[docId][i];
					
					double[] prob = new double[numVmf];
					double maxLogProb = - Double.MAX_VALUE;
					for (int h = 0; h < numVmf; ++h) {
						double[] vecSum = new double[embLen];
						for (int j = 0; j < embLen; ++j) 
							vecSum[j] = kappasVmf[h] * sumXs[h][j] + C0 * mu0[j];
						double lengthExc = VecOp.getL2(vecSum);
	//					System.out.println("length_exc_" + h + "=" + lengthExc);
						for (int j = 0; j < embLen; ++j) 
							vecSum[j] += kappasVmf[h] * x[xIds[docId][i]][j] * w[docId][i];
						double lengthInc = VecOp.getL2(vecSum);
	//					System.out.println("length_inc_" + h + "=" + lengthInc);
						prob[h] = Math.log(alpha + sumZs[docId][h])
								+ vmfLogConstant(kappasVmf[h])
								+ calcLogVmfConstantQuotient(lengthExc, lengthInc);
						if (prob[h] > maxLogProb) maxLogProb = prob[h];
					}
	//				for (int h = 0; h < numVmf; ++h)
	//					System.out.println("z_cond_prob_" + h + "=" + prob[h]);
					
					for (int h = 0; h < numVmf; ++h) prob[h] = Math.exp(prob[h] - maxLogProb);
					prob = VecOp.vec2Dist(prob);
					int newZi = VecOp.drawFromCatDist(prob);
					
					z[docId][i] = newZi;
					for (int j = 0; j < embLen; ++j) sumXs[z[docId][i]][j] += x[xIds[docId][i]][j] * w[docId][i];
					sumZs[docId][z[docId][i]] += 1.0 * w[docId][i];
					sumZOverDocs[z[docId][i]] += 1.0 * w[docId][i];
				}
			}
			
			// Sample kappa_h's
			double avgAccRate = 0.0;
			for (int h = 0; h < numVmf; ++h) {
				double kappaCur = kappasVmf[h];
				
				// Metropolis
				int acc = 0, tot = 0;
				for (int kappaIter = 0; kappaIter < this.kappaGibbsIter; ++kappaIter) {
					double kappaNext = proposeKappa(kappaCur);
					double logPiCur = calcLogKappaPosterior(kappaCur, sumZOverDocs[h], sumXs[h]);
					double logPiNext = calcLogKappaPosterior(kappaNext, sumZOverDocs[h], sumXs[h]);
					double r = Math.exp(logPiNext - logPiCur);
					if (rand.nextDouble() <= r) {
						kappaCur = kappaNext;
						++acc;
					}
					++tot;
				}
				avgAccRate += (double) acc / tot;
				kappasVmf[h] = kappaCur;
			}
			avgAccRate /= numVmf;
			System.out.println("[Vmf Mixture] Kappa acceptance rate: " + avgAccRate);
			
			// Push this sample
			for (int docId = 0; docId < nd; ++docId) {
				double[] distTemp = new double[numVmf];
				for (int h = 0; h < numVmf; ++h) distTemp[h] = alpha + sumZs[docId][h];
				this.distVmf[docId] = VecOp.vec2Dist(distTemp);
			}

			for (int h = 0; h < numVmf; ++h) {
				this.musVmf[h] = VecOp.normalize(sumXs[h]);
			}
			
			double[] aTemp = new double[numVmf];
			for (int docId = 0; docId < nd; ++docId) {
				for (int h = 0; h < numVmf; ++h) aTemp[h] += distVmf[docId][h];
			}
			aTemp = VecOp.vec2Dist(aTemp);

//			for (int h = 0; h < numVmf; ++h)
//				System.out.println("[Vmf Mixture]kappa_" + h + "=" + kappasVmf[h] + ", alpha_" + h + "=" + aTemp[h]);
			
//			double L = calcLogLikelihood(x, xIds, w);
//			System.out.println("[Vmf Mixture] Gibbs L = " + L);
		}
		
		// Use the last sample from posterior as the estimate
	}
	
	private double calcLogKappaPosterior(double kappa, double sumZ, double[] sumX) {
		double[] tempVec = new double[embLen];
		for (int j = 0; j < embLen; ++j) 
			tempVec[j] = kappa * sumX[j] + C0 * mu0[j];
		double logPi = logNormal.logDensity(kappa)
				     + vmfLogConstant(C0)
				     + calcLogVmfConstantQuotientWithNominatorPow(kappa, sumZ, VecOp.getL2(tempVec));
		return logPi;
	}
	
	/**
	 * Calculate the Log Likelihood from the current model
	 * @param x  Dictionary of all possible vectors
	 * @param xIds  xIds[i][j] is an integer, indicating the vector id of the j-th word of the i-th document
	 * @param w  w[i][j] is the weight for the j-th word in the i-th document
	 */
	public double calcLogLikelihood(double[][] x, int[][] xIds, double[][] w) {
		// Calculate Likelihood
		double L = 0.0;
		for (int docId = 0; docId < xIds.length; ++docId) {
			for (int i = 0; i < xIds[docId].length; ++i) {
				double pXi = 0.0;
				double[] tempDist = new double[numVmf];
				double maxL = -Double.MAX_VALUE;
				for (int h = 0; h < numVmf; ++h) {
					double l = 0.0;
					l += Math.log(distVmf[docId][h]);
					l += vmfLogLikelihood(musVmf[h], kappasVmf[h], x[xIds[docId][i]]);
					tempDist[h] = l;
					maxL = maxL > tempDist[h] ? maxL : tempDist[h];
				}
				for (int h = 0; h < numVmf; ++h) pXi += Math.exp(tempDist[h] - maxL);
				pXi = Math.log(pXi) + maxL;
				L += w[docId][i] * pXi;
			}
		}
		return L;
	}
	
	
	private void doJob(String[] args) throws Exception {
		// Init word embedding
        double[][] fullD = new double[TextDataModelAsWordEmbVecs.termList.size()][TextDataModelAsWordEmbVecs.embLen];
		for (int i = 0; i < TextDataModelAsWordEmbVecs.termList.size(); ++i) {
			TermDataEmbModel term = TextDataModelAsWordEmbVecs.termList.get(i);	
			for (int j = 0; j < fullD[i].length; ++j) fullD[i][j] = term.v[j];
			fullD[i] = VecOp.normalize(fullD[i]);
		}
		
		// Read data set
		DocumentDataSet dataSet = new DocumentDataSet();
		dataSet.readFromFile(corpusFileName, TextDataModelAsWordEmbVecs.class);
		
		// Construct input parameters
		int[][] xIds = new int[dataSet.docList.size()][];
		double[][] w = new double[dataSet.docList.size()][]; 
        for (int docId = 0; docId < dataSet.docList.size(); ++docId) {
        	DocumentDataModel doc = dataSet.docList.get(docId);
        	TextDataModelAsWordEmbVecs  text = (TextDataModelAsWordEmbVecs)(doc.text);
        	List<Integer> xIdList = new ArrayList<Integer>();
        	List<Double> wList = new ArrayList<Double>();
        	for (Entry<Integer, Integer> entry : text.tid2Cnt.entrySet()) {
        		for (int j = 0; j < entry.getValue(); ++j) {
        			xIdList.add(entry.getKey());
        			wList.add(1.0);
        		}
        	}
        	xIds[docId] = new int[xIdList.size()];
        	w[docId] = new double[wList.size()];
        	for (int i = 0; i < xIdList.size(); ++i) {
        		xIds[docId][i] = xIdList.get(i);
        		w[docId][i] = wList.get(i);
        	}
        }

        // Infer Embedded vMF Allocation
		initVmfTopicModel();
		inferVmfMixtureByGibbsSampling(fullD, xIds, w);
		
		// Save model
		saveModel("model.vmfs");
	}
	
	/**
	 * Load parameters from an existing vmfs file
	 * @param fileName
	 * @throws Exception
	 */
	public void loadModel(String fileName) throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(new File(fileName)));
		String s = br.readLine();
		String[] slist = s.split("\\t");
		this.numVmf = Integer.parseInt(slist[0]);
		this.embLen = Integer.parseInt(slist[1]);
		int docNum = Integer.parseInt(slist[2]);
		kappasVmf = new double[numVmf];
		musVmf = new double[numVmf][embLen];
		distVmf = new double[docNum][numVmf];
		for (int h = 0; h < numVmf; ++h) {
			s = br.readLine();
			kappasVmf[h] = Double.parseDouble(s);
			s = br.readLine();
			slist = s.split("\\t");
			for (int j = 0; j < embLen; ++j) {
				musVmf[h][j] = Double.parseDouble(slist[j]);
			}
		}
		for (int docId = 0; docId < docNum; ++docId) {
			s = br.readLine();
			slist = s.split("\\t");
			for (int h = 0; h < numVmf; ++h) {
				distVmf[docId][h] = Double.parseDouble(slist[h]);
			}
		}
		br.close();
	}
	
	/**
	 * Save the current model into a file
	 * @param fileName
	 * @throws Exception
	 */
	public void saveModel(String fileName) throws Exception {
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(fileName)));
		bw.write(numVmf + "\t" + embLen + "\t" + distVmf.length + "\n");
	 		for (int h = 0; h < numVmf; ++h) {
				bw.write(kappasVmf[h] + "\n");
				for (int j = 0; j < embLen; ++j) {
					bw.write(musVmf[h][j] + "\t");
				}
				bw.write("\n");
			}
	 		for (int docId = 0; docId < distVmf.length; ++docId) {
	 			for (int h = 0; h < numVmf; ++h) {
	 				bw.write(distVmf[docId][h] + "\t");
	 			}
	 			bw.write("\n");
	 		}
		bw.close();
	}
	
	
	static public void main(String[] args) throws Exception {
		TextDataModelAsWordEmbVecs.init(args);
		VmfTopicModel vmfTopicModel = new VmfTopicModel(args);
		vmfTopicModel.doJob(args);
	}


}
