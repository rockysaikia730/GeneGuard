import math
from Bio.Blast import NCBIWWW, NCBIXML

# ---------- Helper Functions ----------
def evalue_to_score(evalue, cap=1e-180):
    ev = max(evalue, 1e-300)
    s = -math.log10(ev) / (-math.log10(cap))
    return max(0.0, min(1.0, s))

def identity_to_score(identities, align_len):
    if align_len == 0:
        return 0.0
    frac = identities / align_len
    return max(0.0, min(1.0, frac))  

def coverage_to_score(align_len, query_len):
    if query_len == 0:
        return 0.0
    cov = align_len / query_len
    return max(0.0, min(1.0, cov))

# ---------- Simple taxonomy / keyword flagging ----------
dangerous_keywords = [
    "virus", "phage", "bacteriophage", "anthracis", "yersinia", "tox", "toxin", 
    "toxin gene", "virulence", "plasmid", "antibiotic resistance", "resistance"
]

def taxon_risk_score(hit_title):
    title_low = hit_title.lower()
    for kw in dangerous_keywords:
        if kw in title_low:
            return 1.0
    return 0.0

# ---------- Aggregate across alignments/hsp ----------
def blast_record_risk_score(blast_record, top_hits=10, weights=(0.45, 0.25, 0.2, 0.1)):
    """
    returns overall risk score in [0,1] and per-hit breakdown
    weights: identity, coverage, evalue, taxonomy
    """
    qlen = blast_record.query_length if hasattr(blast_record, 'query_length') else None
    if qlen is None:
        qlen = getattr(blast_record, "query_length", 0)

    per_hit_scores = []
    for aln in blast_record.alignments[:top_hits]:
        # take the top HSP for this alignment
        if not aln.hsps:
            continue
        hsp = aln.hsps[0]

        # % of bases matched
        id_s = identity_to_score(hsp.identities, hsp.align_length)
        # % of query that is aligned
        cov_s = coverage_to_score(hsp.align_length, qlen or hsp.align_length)
        # Checks the goodness of E value 
        ev_s = evalue_to_score(hsp.expect)
        # Checks risky keywords
        tax_s = taxon_risk_score(aln.title)
        # combine per the weights
        combined = weights[0]*id_s + weights[1]*cov_s + weights[2]*ev_s + weights[3]*tax_s
        
        per_hit_scores.append({
            "title": aln.title,
            "accession": getattr(aln, "accession", ""),
            "identity": id_s,
            "coverage": cov_s,
            "evalue_score": ev_s,
            "taxon_score": tax_s,
            "combined": combined,
            "raw_evalue": hsp.expect,
            "align_len": hsp.align_length
        })

    # overall risk: take max of per-hit combined (conservative) or a weighted mean
    overall_max = max((h["combined"] for h in per_hit_scores), default=0.0)
    # final conservative score 
    final_score = overall_max
    # Ensure Risk is in [0,1]
    final_score = min(max(final_score, 0.0), 1.0)

    return final_score, per_hit_scores

def get_risk_score(sequence, top_hits=5):
    result_handle = NCBIWWW.qblast(
        program="blastn",
        database="nt",
        sequence=sequence
    )

    blast_record = NCBIXML.read(result_handle)
    score, hits = blast_record_risk_score(blast_record,top_hits)
    return score, hits


if __name__ == "__main__":
    sequence = "AGAAAGTACCGTCTGATGATGTTACCGCTCAGAGATTAATCGTTAGCGGCGGTGAAACAACGTCTTCAGCAGATGGTGCAATGATAACGTTGCATGGTTCCGGAAGCAGTACTCCACGTCGCGCGGTATATAACGCACTCGAACATCTTTTTGAGAACGGAGATGTTAAACCTTATCTTGATAATGTAAATGCTCTTGGTGGTCCGGGAAACAGGTTCTCGACAGTTTATCTTGGCTCCAATCCTGTGGTTACCAGTGACGGAACATTAAAGACAGAGCCGGTCTCTCCTGACGAAGCATTGCTGGATGCCGGGGGTGACGTCAGGTATATCGCTTATAAATGGCTGAACGCTGTCGCTATAAAGGGGGAAGAAGGGGCGAGGATACATCATGGTGTAATCGCGCAGCAACTTCGTGATGTTCTTATTTCTCACGGACTCATGGAAGAAGAAAGCACAACATGCCGCTATGCCTTTCTTTGCTATGACGATTATCCCGCA"
    score, hits = get_risk_score(sequence)
