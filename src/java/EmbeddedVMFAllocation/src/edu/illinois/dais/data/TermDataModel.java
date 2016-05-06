package edu.illinois.dais.data;

/**
 * Storing data of a term (either uni-gram or a phrase).
 * @author hzhuang
 *
 */
public class TermDataModel {
	public int id;
	public String name;
	
	@Override
	public String toString() { 
		return name;
	}
}
