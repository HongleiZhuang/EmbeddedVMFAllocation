package edu.illinois.dais.data;

/**
 * Abstract class for any text fragments, e.g. sentences, paragraphs, or documents.
 * @author hzhuang
 *
 */
public abstract class TextDataModel {
	public String content;
	
	@Override
	public String toString() {
		return content;
	}
}
