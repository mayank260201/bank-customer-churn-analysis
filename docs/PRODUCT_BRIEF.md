# Product Brief — Churn Early-Warning for a Retail Bank

> This is the "product" side of the project: if this model became a real tool inside a bank, **who would use it, what problem it solves, how we'd measure success, and what could go wrong.**
>
> Labels used: **[Fact]** = from the data/model in this repo · **[Assumption]** = my guess, not in the data · **[Idea]** = a proposal

---

## 1. Problem

- **[Fact]** 20.4% of the bank's 10,000 customers left (2,037 customers).
- **[Fact]** Those customers took **24.3% of total deposits (balance)** with them, because churners hold *more* money on average than customers who stay.
- **[Interpretation]** Today the bank only knows a customer left *after* they close the account. By then it's too late to do anything.

**Problem statement:** *The retention team doesn't know which customers are about to leave, so it can't act in time and loses high-value deposits.*

## 2. Users

| User | What they need |
|---|---|
| **Retention / Relationship Manager** (main user) | A short list every month: *who to call first and why* |
| **Branch / Regional Head** | Where churn is concentrated (country, age, product) |
| **Product team** | Which products/features are linked to leaving (e.g. 3–4 product bundles) |

**User stories**
- As a relationship manager, I want a ranked call list, so I spend my limited calls on customers most likely to leave *and* with the most money.
- As a regional head, I want churn by country and segment, so I can see where to focus.
- As a product manager, I want to know which product combinations are linked to churn, so I can fix the offer.

## 3. What the data says (key insights)

| # | Insight | Number | Possible action **[Idea]** |
|---|---|---|---|
| 1 | Germany churns 2× more | 32.4% vs ~16% | Investigate Germany: pricing? service? competitor? |
| 2 | Age 50–59 is the riskiest age | 56.0% churn | Pre-retirement / wealth-planning offers |
| 3 | 2 products = most loyal | 7.6% churn (1 product: 27.7%) | Cross-sell a *second* product to 1-product customers |
| 4 | 3–4 products → almost everyone leaves | 82.7% / 100% | Review these bundles — something is wrong (mis-selling? fees?) |
| 5 | Inactive members churn 2× | 26.9% vs 14.3% | Re-engagement campaign (app nudges, RM check-in) |
| 6 | Inactive + Germany is the worst segment | 41.1% | First segment to pilot |
| 7 | Women churn more | 25.1% vs 16.5% | Research *why* (survey) — do **not** target by gender (see Risks) |

## 4. The solution (what this repo builds)

```
Customer data ──► ML model (Random Forest) ──► churn probability per customer
                                                   │
                       ┌───────────────────────────┼─────────────────────────┐
                       ▼                           ▼                         ▼
              Risk band (High/Med/Low)    Retention call list        Power BI dashboard
                                          (SQL view + Excel)         (for managers)
```

- **[Fact]** Model: ROC-AUC **0.854** on test data. At threshold 0.3 it catches **~67% of churners** (precision 57%).
- **[Fact]** The risk bands work: in the **High** band 83.8% really churned; in **Low** only 8.9%.

### Prioritisation — the key product decision
The retention team can't call everyone. **[Assumption]** say they can make ~1,000 calls a month.

| Who to call (1,000 calls) | Real churners reached | % of churned balance reached |
|---|---|---|
| Random 1,000 customers | ~211 | ~10% |
| Top 1,000 by **churn probability** | 825 | 41% |
| Top 1,000 by **probability × balance** | 731 | **51%** |

**[Fact]** (from `sql/05_model_results.sql` Q6)
**Decision:** rank by **probability × balance** if the goal is *protecting deposits*; rank by probability if the goal is *number of customers kept*. This is a business choice, not a technical one — I'd confirm it with the retention head.

## 5. Success metrics

| Type | Metric | Target **[Assumption]** |
|---|---|---|
| North-star | **Deposit balance retained** from contacted customers | ↑ vs control group |
| Outcome | Churn rate of contacted High-risk customers vs a non-contacted control group | −10 to −20% relative |
| Model | Recall of churners in top-N list; precision of High band | Recall ≥ 60%, High-band precision ≥ 75% |
| Adoption | % of call list actually contacted by RMs each month | ≥ 80% |
| Guardrail | Complaints / opt-outs from retention calls | No increase |

**How to prove it works:** A/B test — split High-risk customers randomly: half get the retention call, half don't. Compare churn after 3 months. Without a control group we can't say the calls caused anything.

## 6. Simple value estimate **[Assumption-heavy]**

If 1,000 calls reach 731 real churners and **[Assumption]** 20% of them are convinced to stay:
- ~146 customers saved per month
- **[Fact]** the churners in that list hold on average ~129K balance each
- → roughly **18.8M of deposits kept per month** (dataset doesn't give the currency)

This is only a rough sizing to decide if the idea is worth piloting. The real number comes from the A/B test.

## 7. Risks & limits

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Fairness** – model uses gender and age | Treating customers differently by gender can be discriminatory / against regulations | Use model only to *prioritise outreach*, never for pricing/credit decisions; test a version without gender |
| **Snapshot data** – no dates | Can't tell *when* someone will leave; model learned from one point in time | Next version: monthly data, predict "churn in next 90 days" |
| **False alarms** (43% of flagged are not churners) | Wasted calls, may annoy loyal customers | Calls should be helpful (service check-in), not "please don't leave" |
| **Model drift** | Customer behaviour changes | Re-score monthly; re-train every quarter; monitor recall |
| **Correlation ≠ cause** | "3–4 products → churn" may be mis-selling, not the products | Talk to RMs + customers before changing products |

## 8. Roadmap **[Idea]**

| Version | What | Why |
|---|---|---|
| **v1 (this repo)** | Batch scoring (`predict.py`), SQL call list, Power BI | Prove value with low effort |
| v2 | Monthly automatic re-scoring from the database + email list to RMs | Remove manual work |
| v3 | Show "top 3 reasons" per customer (e.g. inactive, age, Germany) in the CRM | RMs trust and act on it more |
| v4 | A/B tested retention offers per segment | Learn which action works for whom |

## 9. What I would ask stakeholders first
1. Is the goal to keep **customers** or **deposits**? (changes the ranking)
2. How many retention calls can the team make per month?
3. What does a retention offer cost? (sets the threshold)
4. Are we allowed to use age/gender in the model? (compliance)
