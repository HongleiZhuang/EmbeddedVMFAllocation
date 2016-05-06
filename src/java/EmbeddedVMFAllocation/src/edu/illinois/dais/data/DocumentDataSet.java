package edu.illinois.dais.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;


public class DocumentDataSet {
	
	public List<DocumentDataModel> docList = new ArrayList<DocumentDataModel>();
	
	public String dataSetName;
	
	private Class<? extends TextDataModel> textType = null;
	
	public Class<? extends TextDataModel> getTextType() {
		return textType;
	}
	
	private long addDoc(DocumentDataModel doc) {
		doc.id = this.docList.size();
		this.docList.add(doc);
		return doc.id;
	}
	
	public void readFromFile(String fileName, Class<? extends TextDataModel> textType) throws Exception {
		this.textType = textType;
		BufferedReader br = new BufferedReader(new FileReader(new File(fileName)));
		String s;

		while ((s = br.readLine()) != null) {
			s = s.trim();
			DocumentDataModel doc = new DocumentDataModel();
			
			String content = s;
			TextDataModel text = null;
			if (textType.equals(TextDataModelAsWordEmbVecs.class)) {
				text = TextDataModelAsWordEmbVecs.createTextDataModelAsWordEmbVecs(content);
			}

			doc.text = text;
			doc.fileName = "line_" + docList.size();
//			if (VecOp.getL1(text.tid2Cnt) < 50) continue; // TODO: Remove too short documents.  
			addDoc(doc);
			if (docList.size() % 100 == 0) System.out.println("[Reading documents] " + docList.size() + " processed...");
		}
		br.close();
		System.out.println("[Reading documents] Done.");
	}

 }
