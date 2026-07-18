from __future__ import annotations

from itertools import cycle
from typing import Any

DOMAINS = ["recruitment", "sales", "support", "finance", "incident"]
DATASET_VERSION = "v0.1.0"


def _base_record(index: int, task: str, domain: str, difficulty: str, tags: list[str], synthetic: bool) -> dict[str, Any]:
    return {
        "id": f"{task.split('_')[0]}_{index:04d}",
        "task": task,
        "domain": domain,
        "difficulty": difficulty,
        "source_type": "synthetic" if synthetic else "human_authored",
        "source_name": "project_template_generator" if synthetic else "project_authored_seed",
        "license": "project_owned",
        "contains_pii": False,
        "is_synthetic": synthetic,
        "creation_method": "controlled_template_variation",
        "review_status": "approved",
        "duplicate_group_id": None,
        "dataset_version": DATASET_VERSION,
        "tags": tags,
    }


def business_rewriting(index: int, synthetic: bool = True) -> dict[str, Any]:
    situations = [
        ("資料見たら返事ください", "取引先への丁寧な依頼文に書き換えてください。", "お手数をおかけいたしますが、資料をご確認のうえ、ご返信いただけますと幸いです。"),
        ("明日の打ち合わせ、無理なら言って", "上司への丁寧な確認文に書き換えてください。", "明日の打ち合わせにつきまして、ご都合が難しい場合はお知らせいただけますでしょうか。"),
        ("請求書まだ?", "経理担当者への失礼のない確認文にしてください。", "請求書のご送付状況につきまして、念のため確認させていただけますでしょうか。"),
        ("納期遅れます。すみません。", "顧客向けのお詫びと共有文に書き換えてください。", "納期に遅れが生じる見込みとなり、誠に申し訳ございません。状況と今後の対応について速やかにご共有いたします。"),
    ]
    raw, instruction, reference = situations[index % len(situations)]
    record = _base_record(index, "business_rewriting", DOMAINS[index % len(DOMAINS)], "medium", ["keigo", "business"], synthetic)
    record.update({"input": f"{raw}（案件番号{index:03d}）", "instruction": instruction, "reference_answer": reference})
    return record


def information_extraction(index: int, synthetic: bool = True) -> dict[str, Any]:
    companies = ["Example株式会社", "東都システムズ", "青葉物流", "未来バイオ"]
    dates = ["2026-08-03", "2026-09-14", "2026-10-22", "2026-11-05"]
    times = ["14:00", "10:30", "16:00", "09:00"]
    company = companies[index % len(companies)]
    date = dates[index % len(dates)]
    time = times[index % len(times)]
    input_text = f"{company}より、{date} {time}からオンラインで採用面談を実施したいとの連絡。参加可否を本日中に返信する必要がある。管理番号{index:04d}。"
    reference = {
        "company": company,
        "meeting_date": date,
        "meeting_time": time,
        "meeting_type": "オンライン",
        "required_action": "参加可否を返信",
    }
    record = _base_record(index, "information_extraction", DOMAINS[index % len(DOMAINS)], "medium", ["schema", "extraction"], synthetic)
    record.update(
        {
            "input": input_text,
            "instruction": "本文から会議情報をJSONで抽出してください。",
            "reference_answer": reference,
            "expected_schema": {key: "string" for key in reference},
        }
    )
    return record


def summarization(index: int, synthetic: bool = True) -> dict[str, Any]:
    owners = ["佐藤", "田中", "鈴木", "高橋"]
    owner = owners[index % len(owners)]
    deadline = ["2026-08-10", "2026-09-20", "2026-10-31", "2026-11-15"][index % 4]
    input_text = (
        f"定例会議では、法人向けFAQの回答品質が部署ごとにばらつく点が課題になった。"
        f"{owner}さんが評価基準を整理し、{deadline}までにレビュー表を共有する。"
        f"次回は問い合わせ分類と回答根拠の表示方法を確認する。議事録番号{index:04d}。"
    )
    reference = {
        "summary": "法人向けFAQの回答品質を揃えるため、評価基準を整理する。",
        "decisions": ["部署共通のレビュー表を作成する"],
        "action_items": [{"owner": owner, "task": "評価基準を整理してレビュー表を共有", "deadline": deadline}],
    }
    record = _base_record(index, "summarization", "support", "medium", ["summary", "action_items"], synthetic)
    record.update(
        {
            "input": input_text,
            "instruction": "要約、決定事項、アクションアイテムをJSONで出力してください。",
            "reference_answer": reference,
            "expected_schema": {"summary": "string", "decisions": "array", "action_items": "array"},
        }
    )
    return record


def grounded_qa(index: int, synthetic: bool = True) -> dict[str, Any]:
    contexts = [
        ("経費精算規程では、交通費は利用日から30日以内に領収書を添付して申請する。宿泊費は事前承認が必要である。", "交通費はいつまでに申請する必要がありますか。", "利用日から30日以内に申請する必要があります。"),
        ("製品保証は購入日から1年間有効で、消耗品と不適切な改造による故障は対象外である。", "消耗品は保証対象ですか。", "いいえ、消耗品は保証対象外です。"),
        ("社内AI利用方針では、個人情報を含む顧客データを外部サービスへ入力してはならない。", "顧客の個人情報を外部AIサービスに入力できますか。", "いいえ、入力してはなりません。"),
        ("リモート勤務規程では、週2日まで在宅勤務を申請できる。海外からの勤務については規程に記載がない。", "海外から勤務できますか。", "本文には海外からの勤務可否は記載されていません。"),
    ]
    context, question, answer = contexts[index % len(contexts)]
    record = _base_record(index, "grounded_qa", "policy", "hard" if "記載がない" in context else "medium", ["grounding", "abstention"], synthetic)
    record.update(
        {
            "context": context,
            "input": f"{question} 確認番号{index:04d}。",
            "instruction": "与えられた文書だけを根拠に回答してください。根拠がない場合は不明と述べてください。",
            "reference_answer": answer,
        }
    )
    return record


def robustness(index: int, synthetic: bool = True) -> dict[str, Any]:
    inputs = [
        ("令和8年7月18日に3,000,000円の発注を受けた。", "日付と金額を正規化してください。", {"date": "2026年7月18日", "amount_jpy": "3000000"}),
        ("ＡＰＩ　レスポンスが　５００　になりました", "全角文字と余分な空白を正規化してください。", "API レスポンスが 500 になりました"),
        ("ミーティングわ明日で、担当わ佐藤です。", "誤字を修正し、自然な業務文にしてください。", "ミーティングは明日で、担当は佐藤です。"),
        ("三百万円の見積もりと300万円の請求額に差分はありますか。", "金額表記を解釈して差分有無を答えてください。", "差分はありません。どちらも300万円を表しています。"),
    ]
    input_text, instruction, reference = inputs[index % len(inputs)]
    record = _base_record(index, "robustness", "operations", "hard", ["normalization", "mixed_script"], synthetic)
    record.update({"input": f"{input_text} ケース{index:03d}", "instruction": instruction, "reference_answer": reference})
    return record


TASK_GENERATORS = [
    business_rewriting,
    information_extraction,
    summarization,
    grounded_qa,
    robustness,
]


def generate_template_records(count: int, *, synthetic: bool, start_index: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    counters = {generator.__name__: start_index for generator in TASK_GENERATORS}
    for generator in cycle(TASK_GENERATORS):
        if len(records) >= count:
            break
        counters[generator.__name__] += 1
        records.append(generator(counters[generator.__name__], synthetic=synthetic))
    return records
