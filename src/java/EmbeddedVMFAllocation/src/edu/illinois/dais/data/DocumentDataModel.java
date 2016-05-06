package edu.illinois.dais.data;

public class DocumentDataModel {
	public long id;
	public TextDataModel text;
	public String fileName = "*unknown*";  //optional
	
	@Override
	public String toString() {
		String ret = "";
		ret += this.id + "\t" + this.fileName + "\n";
		ret += text.toString() + "\n";
		return ret;
	}
}
