package edu.illinois.dais.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


/**
 * Representing text as a bag of embedded vectors.
 * Needs to be initialized before usage by providing the word embedding file.
 * @author hzhuang
 *
 */
public class TextDataModelAsWordEmbVecs extends TextDataModel {
	final Set<String> stopWords = new HashSet<String>(Arrays.asList(
			"a\'s", "able", "about", "above", "according",
			"accordingly", "across", "actually", "after", "afterwards",
			"again", "against", "ain\'t", "all", "allow",
			"allows", "almost", "alone", "along", "already",
			"also", "although", "always", "am", "among",
			"amongst", "an", "and", "another", "any",
			"anybody", "anyhow", "anyone", "anything", "anyway",
			"anyways", "anywhere", "apart", "appear", "appreciate",
			"appropriate", "are", "aren\'t", "around", "as",
			"aside", "ask", "asking", "associated", "at",
			"available", "away", "awfully", "be", "became",
			"because", "become", "becomes", "becoming", "been",
			"before", "beforehand", "behind", "being", "believe",
			"below", "beside", "besides", "best", "better",
			"between", "beyond", "both", "brief", "but",
			"by", "c\'mon", "c\'s", "came", "can",
			"can\'t", "cannot", "cant", "cause", "causes",
			"certain", "certainly", "changes", "clearly", "co",
			"com", "come", "comes", "concerning", "consequently",
			"consider", "considering", "contain", "containing", "contains",
			"corresponding", "could", "couldn\'t", "course", "currently",
			"definitely", "described", "despite", "did", "didn\'t",
			"different", "do", "does", "doesn\'t", "doing",
			"don\'t", "done", "down", "downwards", "during",
			"each", "edu", "eg", "eight", "either",
			"else", "elsewhere", "enough", "entirely", "especially",
			"et", "etc", "even", "ever", "every",
			"everybody", "everyone", "everything", "everywhere", "ex",
			"exactly", "example", "except", "far", "few",
			"fifth", "first", "five", "followed", "following",
			"follows", "for", "former", "formerly", "forth",
			"four", "from", "further", "furthermore", "get",
			"gets", "getting", "given", "gives", "go",
			"goes", "going", "gone", "got", "gotten",
			"greetings", "had", "hadn\'t", "happens", "hardly",
			"has", "hasn\'t", "have", "haven\'t", "having",
			"he", "he\'s", "hello", "help", "hence",
			"her", "here", "here\'s", "hereafter", "hereby",
			"herein", "hereupon", "hers", "herself", "hi",
			"him", "himself", "his", "hither", "hopefully",
			"how", "howbeit", "however", "i\'d", "i\'ll",
			"i\'m", "i\'ve", "ie", "if", "ignored",
			"immediate", "in", "inasmuch", "inc", "indeed",
			"indicate", "indicated", "indicates", "inner", "insofar",
			"instead", "into", "inward", "is", "isn\'t",
			"it", "it\'d", "it\'ll", "it\'s", "its",
			"itself", "just", "keep", "keeps", "kept",
			"know", "known", "knows", "last", "lately",
			"later", "latter", "latterly", "least", "less",
			"lest", "let", "let\'s", "like", "liked",
			"likely", "little", "look", "looking", "looks",
			"ltd", "mainly", "many", "may", "maybe",
			"me", "mean", "meanwhile", "merely", "might",
			"more", "moreover", "most", "mostly", "much",
			"must", "my", "myself", "name", "namely",
			"nd", "near", "nearly", "necessary", "need",
			"needs", "neither", "never", "nevertheless", "new",
			"next", "nine", "no", "nobody", "non",
			"none", "noone", "nor", "normally", "not",
			"nothing", "novel", "now", "nowhere", "obviously",
			"of", "off", "often", "oh", "ok",
			"okay", "old", "on", "once", "one",
			"ones", "only", "onto", "or", "other",
			"others", "otherwise", "ought", "our", "ours",
			"ourselves", "out", "outside", "over", "overall",
			"own", "particular", "particularly", "per", "perhaps",
			"placed", "please", "plus", "possible", "presumably",
			"probably", "provides", "que", "quite", "qv",
			"rather", "rd", "re", "really", "reasonably",
			"regarding", "regardless", "regards", "relatively", "respectively",
			"right", "said", "same", "saw", "say",
			"saying", "says", "second", "secondly", "see",
			"seeing", "seem", "seemed", "seeming", "seems",
			"seen", "self", "selves", "sensible", "sent",
			"serious", "seriously", "seven", "several", "shall",
			"she", "should", "shouldn\'t", "since", "six",
			"so", "some", "somebody", "somehow", "someone",
			"something", "sometime", "sometimes", "somewhat", "somewhere",
			"soon", "sorry", "specified", "specify", "specifying",
			"still", "sub", "such", "sup", "sure",
			"t\'s", "take", "taken", "tell", "tends",
			"th", "than", "thank", "thanks", "thanx",
			"that", "that\'s", "thats", "the", "their",
			"theirs", "them", "themselves", "then", "thence",
			"there", "there\'s", "thereafter", "thereby", "therefore",
			"therein", "theres", "thereupon", "these", "they",
			"they\'d", "they\'ll", "they\'re", "they\'ve", "think",
			"third", "this", "thorough", "thoroughly", "those",
			"though", "three", "through", "throughout", "thru",
			"thus", "to", "together", "too", "took",
			"toward", "towards", "tried", "tries", "truly",
			"try", "trying", "twice", "two", "un",
			"under", "unfortunately", "unless", "unlikely", "until",
			"unto", "up", "upon", "us", "use",
			"used", "useful", "uses", "using", "usually",
			"value", "various", "very", "via", "viz",
			"vs", "want", "wants", "was", "wasn\'t",
			"way", "we", "we\'d", "we\'ll", "we\'re",
			"we\'ve", "welcome", "well", "went", "were",
			"weren\'t", "what", "what\'s", "whatever", "when",
			"whence", "whenever", "where", "where\'s", "whereafter",
			"whereas", "whereby", "wherein", "whereupon", "wherever",
			"whether", "which", "while", "whither", "who",
			"who\'s", "whoever", "whole", "whom", "whose",
			"why", "will", "willing", "wish", "with",
			"within", "without", "won\'t", "wonder", "would",
			"wouldn\'t", "yes", "yet", "you", "you\'d",
			"you\'ll", "you\'re", "you\'ve", "your", "yours",
			"yourself", "yourselves", "zero"));
	public Map<Integer, Integer> tid2Cnt;
	
	static public List<TermDataEmbModel> termList = new ArrayList<TermDataEmbModel>();
	static public Map<String, Integer> termName2List = new HashMap<String, Integer>();
	static public int embLen; 
	static public boolean rmStopWords = true;
	
	/**
	 * Parse arguments and initialize accordingly.  
	 * @param args
	 */
	static public void init(String[] args) {
		String wordVecsFileStr = ""; 		
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-wordembs")) {
				wordVecsFileStr = args[++i];
			} else if (args[i].equals("-keepstopword")) {
				rmStopWords = false;
			}
		}
		init(wordVecsFileStr);
	}
	
	/**
	 * Initialize the static dictionary of this representation.
	 * @param wordVecsFileStr
	 */
	static public void init(String wordVecsFileStr){
		// Read embedding
		System.out.println("Read embedding...");
		try {
			BufferedReader br;
			br = new BufferedReader(new FileReader(new File(wordVecsFileStr)));
			String s = br.readLine();
			String[] slist = s.split("\\s+");
			int n = Integer.parseInt(slist[0]); 
			embLen = Integer.parseInt(slist[1]);
			System.out.println("Embedded vector length = " + embLen);
			for (int i = 0; i < n; ++i) {
				s = br.readLine();
				slist = s.split("\\s+");
				double[] v = new double[embLen];
				for (int j = 0; j < embLen; ++j) v[j] = Double.parseDouble(slist[j + 1]);
				addTerm(slist[0], v);
			}
			br.close();
 		} catch (Exception e) {
			System.out.println("[ERROR!] Failed to read word embedding vectors!");
			e.printStackTrace();
		}
	}
	
	static public TextDataModel createTextDataModelAsWordEmbVecs(String content) {
		return new TextDataModelAsWordEmbVecs(content);
	}
	
	private TextDataModelAsWordEmbVecs(String content) {
		this.content = content;
		buildBagOfTerms();
	}
	
	static public Integer getTermId(String termName) {
		termName = termName.toLowerCase(); 
		Integer tid = termName2List.get(termName);
		return tid;
	}
	
	static private int addTerm(String termName, double[] v) {
		termName = termName.toLowerCase(); 
		TermDataEmbModel t = new TermDataEmbModel();
		t.id = termList.size();
		t.name = termName;
		t.v = v;
		termList.add(t);
		termName2List.put(t.name, t.id);
		return t.id;
	}
	
	public void buildBagOfTerms() {
		tid2Cnt = new HashMap<Integer, Integer>();
		String content = this.content;
		content = content.replaceAll("\\.", " . ");
		content = content.replaceAll("\\\"", " \" ");
		content = content.replaceAll(",", " , ");
		content = content.replaceAll("\\(", " ( ");
		content = content.replaceAll("\\)", " ) ");
		content = content.replaceAll("\\!", " ! ");
		content = content.replaceAll("\\?", " ? ");
		content = content.replaceAll("\\;", " ; ");
		content = content.replaceAll("\\:", " : ");
		content = content.replaceAll("\\:", " : ");
		String[] slist = content.split("\\s+");
		for (String word : slist) {
			word = word.toLowerCase();
	        if (rmStopWords && stopWords.contains(word)) continue;
//	        if (word.startsWith("#") || word.startsWith("@") || word.startsWith("RT") || word.startsWith("http://")) continue;
	        if (word.length() < 2) continue;
			Integer tid = getTermId(word);
			if (tid == null) continue;
			tid2Cnt.put(tid, tid2Cnt.containsKey(tid) ? tid2Cnt.get(tid) + 1 : 1);
		}
	}
	
	@Override
	public String toString() {
		String ret = "";
		ret += this.content;
		return ret;
	}
}
