#!/usr/bin/env python3
"""Assemble evaluator-reported KB issues with recorded dialogue and rewrites.

Annotations below are manual review hypotheses, not retrieval ground truth.
The human-facing .json file contains JSONL records with four top-level fields;
available original prompts are preserved as evidence inside analysis.
"""

import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
DATA = ROOT.parent / "data"
CSV = ROOT / "Retell 测试集 R2 - KB key issues.csv"
SYNONYMS = ROOT / "英文同义词表 Copy_数据表_表格.csv"


def issue(response, category, observation, direction, titles=(), limitation=None):
    return {"response_id": str(response), "heavy_issue": True,
            "failure_category": category, "observation": observation,
            "rewrite_experiment": direction, "reference_article_hints": list(titles),
            "limitation": limitation, "causal_status": "review_hypothesis_not_proven"}


# Choose only key failure points, rather than flagging each partial-ASR request.
ANNOTATIONS = {
    "call_5933a98a8b1925ee7f47be4063b": [issue(6, "routing_or_missing_retrieval_evidence",
        "After the X5 manufacturing-date concern, the agent moves directly to Confirm Human Support; the later transfer fails. No retained retrieval log is available.",
        "Investigate the decision to leave the KB-capable path before testing rewriting. Preserve X5, battery manufacturing date January 2025, and the concern about an old purchased battery.",
        limitation="No rewrite request for this call exists in the supplied export. The transfer failure is a separate operational issue.")],
    "call_6d717fabfff7d93935428242d98": [
        issue(17, "terminology_mapping_candidate_with_relevant_retrieval",
            "The caller first describes a black line in the middle of the app image. The logged query preserves that symptom but has no professional fault term; the matching X4 stitching article is already ranked first at this point.",
            "Test stripe/band of screen as a provisional professional term while preserving the literal black-line symptom. Do not add final-export symptoms that the caller has not confirmed yet.",
            ["X Series: Stitching Issues"],
            "This is a terminology-mapping control within the later failed black-line episode, not a demonstrated retrieval failure at this turn. The table is ambiguous and the original retrieval already contains relevant guidance."),
        issue(22, "retrieval_drift_and_lost_product_context",
        "The final black-line query omits X4 and stitching. Response 22 refuses troubleshooting and its returned contexts no longer include the stitching article, although response 17 previously retrieved the exact X4 black-border scenario.",
        "Keep black line in the middle of recorded/exported media and app playback together. Test whether symptom wording can reach stitching guidance; do not invent a confirmed stitching diagnosis.",
        ["X Series: Stitching Issues"],
        "X4 is already absent from this exact input window. A replacement rewriter alone cannot reliably recover it without additional conversation memory.")],
    "call_8aa8efbaa0eb9e64fc059bd2c4a": [issue(4, "warranty_intent_and_retrieval_ranking",
        "The caller asks whether a retailer's extended warranty covers repair. The query phrases it as repair under 'Mr. 360 warranty'; returned contexts are dominated by app extended-warranty registration and cloud benefits. The agent says it lacks policy details.",
        "Distinguish third-party/retailer extended warranty from manufacturer warranty, and keep the damaged-camera repair eligibility question. Do not assume either coverage or exclusion.",
        ["After-Sales Service Policy"],
        "The policy article was present for logged response 3 but not response 4. Whether it actually answers this particular third-party warranty requires review.")],
    "call_33457cb382388d2f0599f40e7a2": [
        issue(22, "retrieval_ranking_or_answer_grounding",
            "After the X3 clarification, the agent instructs 'Sync files to the cloud'. The logged contexts for this same response already include the evaluator's account-binding troubleshooting article, but below subscription-related material.",
            "Retain cloud-media/subscription visibility and X3 context rather than treating this as a new subscription or upload request. Compare ranking and the downstream answer as separate outcomes.",
            ["Insta360+: Unable to View Cloud Media or Subscription Information After Logging in to an Account with an Active Cloud-storage Plan"],
            "The earlier statement that the user already purchased cloud service is outside this exact rewrite input. Relevant guidance was retrieved, so the answer failure is not proof of a retrieval miss."),
        issue(27, "query_drift_after_incorrect_instructions",
            "The query now searches for a missing 'Sync files to the cloud' control, drifting from cloud entitlement/media visibility. Returned contexts emphasize uploads, and the agent escalates when the option cannot be found.",
            "Test preserving the underlying cloud-media visibility problem while describing the unsuccessful X3 instruction. Do not make the missing button the whole problem.",
            ["Insta360+: Unable to View Cloud Media or Subscription Information After Logging in to an Account with an Active Cloud-storage Plan"],
            "The relevant binding article still appears in this retrieval list. Its use by the answer generator also needs investigation.")],
    "call_89bac8aaa7c1f4d637a9f2a6a91": [issue(11, "multi_step_repair_retrieval_drift",
        "After selecting Ace Pro 2 on the repair website, the query asks for next steps. Mail-in form/shipping instructions were retrieved at response 10, but are absent from response 11, which retrieves product/accessory pages and refuses the next step.",
        "Resolve 'And then?' as continuing an Ace Pro 2 mail-in repair request/work order, retaining the preference to send the whole camera instead of using a lens kit.",
        ["A Guide to Insta360 Mail-in Repair Service"],
        "The prior turn already had form/shipping guidance. Conversation memory and response grounding may matter alongside rewriting.")],
    "call_8ca923d434350f6fda924cc22ff": [
        issue(6, "rewrite_drops_platform_constraint",
            "The query 'Which Insta360 app page supports landscape mode?' drops Android tablet even though it remains in the input. Generic playback-page guidance is returned and drives the subsequent unsuccessful directions.",
            "Preserve Android tablet, Insta360 app and landscape-mode compatibility when resolving the follow-up about supported pages.",
            limitation="The evaluator says Android tablets are unsupported; this is an evaluator claim, not a verified conclusion from a compatibility article in the supplied logs."),
        issue(16, "rewrite_drops_device_and_failed_steps",
            "After Direction Lock is disabled and the tablet is rotated, the query becomes 'Cannot use landscape mode on Playback page'. It drops Android tablet and the already attempted steps; the agent repeats the same guidance.",
            "Retain Android tablet and failed landscape-preview steps so compatibility/unsupported-device information can compete with generic playback instructions.",
            limitation="A better query is only a hypothesis until the relevant compatibility knowledge and retrieved results are checked.")],
    "call_a6476bb81e7277b3dbec64447ed": [
        issue(15, "terminology_mapping_candidate_visible_wifi_app_failure",
            "The iPhone can see the X6 Wi-Fi network, but the user reports that the app cannot connect and shows no error. This supports the table's detectable-Wi-Fi/app-connection fault term, before a Wi-Fi connection has been confirmed.",
            "Map the described failure to Can detect WiFi, but unable to connect to mobile APP. Keep visible Wi-Fi distinct from confirmed Wi-Fi connection, and preserve iPhone and X6.",
            ["X Series: Unable to Connect to Insta360 App"],
            "No original rewrite log is supplied for this call. Relevant connection guidance is already retrieved; this is an earlier terminology candidate within the later failed episode, not a confirmed rewriter-caused failure."),
        issue(21, "terminology_mapping_candidate_connected_wifi_app_failure",
            "The caller confirms Connected beneath the camera's Wi-Fi network after reporting app connection failure. The resulting query should retain the app-level failure rather than treat Wi-Fi detection or association as the unresolved problem.",
            "Use Can detect WiFi, but unable to connect to mobile APP with the newly confirmed camera Wi-Fi connection. The brief Yes refers to the preceding connected-network question.",
            ["X Series: Unable to Connect to Insta360 App"],
            "No original rewrite log is supplied for this call. Connection guidance is already present in the retrieval log, so this is a diagnostic-stage mapping candidate rather than a proven failed retrieval."),
        issue(25, "multi_step_connection_retrieval_drift",
            "After the caller opens iPhone Settings for an X6 app-connection problem, response 25 offers transfer. Its retrieval list is dominated by camera settings rather than the phone/app connection workflow.",
            "Keep X6, iPhone, app cannot connect despite camera Wi-Fi connection, and the current troubleshooting step together.",
            ["X Series: Unable to Connect to Insta360 App"],
            "No actual rewrite request is present in this export; the full transcript identifies the point but cannot establish what the rewriter generated."),
        issue(33, "answer_refusal_despite_relevant_context",
            "The caller says the app still cannot connect; response 33 again offers transfer even though its retrieval log includes X Series app-connection solution chunks.",
            "Use as a response-grounding control once rewrite logs are obtained; do not attribute the refusal to rewriting without evidence.",
            ["X Series: Unable to Connect to Insta360 App"],
            "Missing rewrite logs prevent a same-input replay. Relevant context exists in the recorded retrieval list.")],
    "call_97b120465dd8a78feb3f577e283": [issue(3, "routing_before_kb_answer",
        "The caller asks how to find a stolen camera; the next node is Confirm Human Support and the next response offers transfer. Neither rewrite entries nor a retained retrieval log are available for this call.",
        "Investigate routing to the lost/stolen-device FAQ before assuming a rewriter fix. Keep theft/recovery intent; do not promise a tracking feature.",
        ["General FAQs for Insta360 devices"],
        "The call falls inside an exported interval but has no matching rewrite entry. This does not prove that a rewrite was executed or that retrieval returned no results.")],
    "call_57d30883073e0e8a56775624334": [
        issue(8, "kb_scope_or_technical_ranking",
            "The compatibility question is rewritten with X5 intact, but the logged KB and retrieved material are store/product recommendations. The agent lists capacities instead of complete compatibility requirements.",
            "Retain X5 microSD compatibility requirements rather than commercial card recommendations. Compare with the current accurate topic-preserving query as a control.",
            ["X5: Storage Compatibility", "X Series: Memory Card Compatibility FAQ"],
            "The logged search scope is knowledge_base_c113c41c4ee37f7a. A rewriter cannot retrieve a technical article if it is outside that scope; article membership has not been verified."),
        issue(15, "kb_scope_or_ranking_not_obvious_rewrite_error",
            "The query already says 'maximum supported microSD card capacity for X5', but the answer gives 512 GB store options and says the maximum is unconfirmed. Retrieved material includes X5 store pages and an X3 store result.",
            "Use as a control for whether new wording improves the technical-document ranking. Preserve X5 and maximum supported capacity.",
            ["X5: Storage Compatibility", "X Series: Memory Card Compatibility FAQ"],
            "The evaluator states 1 TB; treat this as the reported expected fact, not verified ground truth. The old rewrite already captures the question well.")],
    "call_9bfd490cb5300f581a7747ac74b": [issue(4, "retrieval_ranking_and_ambiguous_service_intent",
        "The query correctly asks about X5 repair benefits, but returned contexts start with lens-kit installation and the answer discusses free lens-kit labor. The caller then explicitly corrects this misunderstanding.",
        "Preserve general repair-service benefits/coverage rather than narrowing to lens installation. Do not infer a specific purchased plan before the caller names one.",
        ["Insta360+: Cloud Service Benefits"],
        "The evaluator cites cloud benefits, while the caller later explicitly names one-year FlexiCare. Cloud-plan entitlement must not be invented; the relevant cloud-benefit title was already in the initial returned list.")],
    "call_3db1ac9d6c2ef35abf1a8026479": [
        issue(16, "upstream_context_window_loses_model",
            "The query becomes 'steps to update camera firmware' and retrieves Link/GO/Ace/X instructions instead of the previously retrieved Luna instructions. The latest input window no longer contains Luna Ultra.",
            "Primary check: retain the camera model upstream in structured memory or context. With the exact existing prompt, do not expect a new rewriter to infer Luna Ultra.",
            ["Luna Series: Check and Update Firmware Version"],
            "This is a context-construction limitation, not demonstrated model omission of an available fact."),
        issue(17, "rewriter_omits_model_available_again",
            "A later snapshot includes the agent's explicit 'Luna Ultra' refusal, yet the generated query is still 'how to update camera firmware'. Luna guidance appears only later in the retrieval list, and the agent repeats its refusal.",
            "Retain Luna Ultra when rewriting the unresolved firmware-update steps request.",
            ["Luna Series: Check and Update Firmware Version"],
            "This snapshot ends with an agent message, not a new user turn; treat it as a secondary re-query experiment. The relevant Luna article was retrieved at a lower position.")],
    "call_3643c8f4b7e75a67c009bcc09ac": [
        issue(3, "rewrite_loses_repair_quote_context",
            "The detailed request to reject an expensive X3 repair quote and return the camera unrepaired is reduced to 'return X3 camera to insta three sixty'. Retrieved chunks concern camera features and troubleshooting rather than the repair-order return process.",
            "Preserve rejection of a repair quote, no authorization/payment for repair, and return of the unrepaired X3. Distinguish this from returning a new purchase.",
            ["After-Sales Service Policy", "A Guide to Insta360 Mail-in Repair Service"],
            "This is an additional failure point visible in the conversation, preceding the evaluator's main warranty/free-repair question."),
        issue(6, "warranty_retrieval_ranking_or_answer_policy",
            "The query correctly asks whether an X3 still under warranty can be repaired for free, but returned chunks emphasize cloud benefits and generic hardware problems. The response again offers transfer instead of explaining applicable coverage conditions.",
            "Retain X3, in-warranty repair eligibility and possible repair charges; search policy rather than unrelated troubleshooting.",
            ["After-Sales Service Policy"],
            "Inspection may legitimately be needed to approve free repair. Do not label a promise of free repair as the desired answer.")],
    "call_e26836d7b834515d8104744c2ad": [issue(5, "theft_lock_intent_vs_camera_feature_results",
        "For the stolen X4 remote-lock request, retrieved chunks include Horizon Lock and camera Wi-Fi rather than theft/blacklisting guidance. The following response offers transfer; the next response's retrieval does find General FAQs and gives police-report advice.",
        "Preserve stolen camera, serial number/proof of purchase and requested remote lock/blacklist, distinguishing this from Horizon Lock or screen lock.",
        ["General FAQs for Insta360 devices"],
        "No rewrite logs for this call are supplied. The exact generated query and whether remote locking is supported cannot be inferred.")],
}


# Proposed queries use only the conversation through the marked user response.
# These are manual review targets, not verified fixes or answers from the KB.
# Category spellings match the requested review-file schema.
REWRITING_CORRECTIONS = {
    "call_33457cb382388d2f0599f40e7a2:user_turn_25": {
        "expected_rewrite": "Insta360 X3 on iPhone: cloud footage is unavailable and the app asks me to subscribe even though I already purchased cloud storage.",
        "correction_reason": "The original query omits the user's earlier statement that cloud storage was already purchased (conversation turn 14), which is outside the logged 10-turn prompt. Restoring that fact with extended history distinguishes an existing-subscription/media-access problem from a new subscription request. The account-binding cause is not assumed; relevant guidance was already retrieved, so answer grounding also needs review.",
        "issure_category": "out_ouf_context",
    },
    "call_33457cb382388d2f0599f40e7a2:user_turn_29": {
        "expected_rewrite": "Insta360 X3 on iPhone: already purchased cloud storage, but cloud footage is unavailable and the app asks me to subscribe; I also cannot find the suggested Sync files to the cloud option on the camera.",
        "correction_reason": "The original searches only for the missing control introduced by the agent and loses the unresolved cloud-access problem. The purchased subscription (turn 14) and subscription prompt (turn 18) are both outside the logged 10-turn input. Extended history should retain that underlying problem while preserving the user's latest report about the missing option; this does not establish that rewriting alone caused the failure.",
        "issure_category": "out_ouf_context",
    },
    "call_3db1ac9d6c2ef35abf1a8026479:user_turn_24": {
        "expected_rewrite": "How do I update the firmware on my Insta360 Luna Ultra camera?",
        "correction_reason": "The generic original omits Luna Ultra and can retrieve firmware instructions for unrelated cameras. Luna Ultra was established earlier (user turn 5, repeated at agent turn 9) but is absent from the logged 10-turn prompt. The proposed query requires that earlier context; the model should not be expected to guess the camera from the short prompt alone.",
        "issure_category": "out_ouf_context",
    },
    "call_6d717fabfff7d93935428242d98:user_turn_27": {
        "expected_rewrite": "Insta360 X4 black line in the middle of recorded media in the app and final exported videos or photos (possible stripe/band of screen); investigate image or stitching artifacts.",
        "correction_reason": "CSV rows 899-900 map black-line/center-of-video descriptions to the exact professional fault term stripe/band of screen. Add that candidate term while retaining the newly confirmed recorded-media and final-export symptoms. Do not add intermittent flickering or a confirmed seam defect: neither is reported, and other table rows map black line differently. The original preserves the symptom but omits this terminology; X4 also requires history beyond the logged 10-turn prompt. This is an overlapping terminology/context experiment, not proof that the mapped fault caused the failure.",
        "issure_category": "terminology_mismatch",
    },
    "call_6d717fabfff7d93935428242d98:user_turn_19": {
        "expected_rewrite": "Insta360 X4 black line in the middle of the image in the Insta360 app (possible stripe/band of screen).",
        "correction_reason": "The original describes the black line literally. CSV rows 899-900 associate black-line and center-of-video wording with the professional fault term stripe/band of screen, which can be included as a provisional retrieval synonym. At this cutoff the caller has not confirmed recorded-media or exported-file symptoms, a seam location or intermittent flickering, so none is asserted. X4 comes from older history. The original retrieval already ranked the X4 stitching article first, making this a mapping control within the later failed episode rather than a proven retrieval miss.",
        "issure_category": "terminology_mismatch",
    },
    "call_a6476bb81e7277b3dbec64447ed:user_turn_17": {
        "expected_rewrite": "Insta360 X6 on iPhone: Can detect WiFi, but unable to connect to mobile APP; camera Wi-Fi is visible, but the Insta360 app cannot connect and shows no error message.",
        "correction_reason": "CSV row 95 maps Can find Wi-Fi but app connection fails to the exact professional fault term Can detect WiFi, but unable to connect to mobile APP. The user's visible-network confirmation and app failure support that distinction instead of Unable to Detect Wi-Fi Signal. Wi-Fi connection is not confirmed until a later turn, so it must not be asserted here. The original rewrite is unavailable; this is a proposed terminology target, not an observed model error, and relevant guidance was already retrieved.",
        "issure_category": "terminology_mismatch",
    },
    "call_a6476bb81e7277b3dbec64447ed:user_turn_25": {
        "expected_rewrite": "Insta360 X6 on iPhone: Can detect WiFi, but unable to connect to mobile APP; the phone confirms connection to camera Wi-Fi, but the Insta360 app cannot connect.",
        "correction_reason": "The user's Yes confirms that the iPhone shows Connected beneath the camera network. Together with the earlier app-connection failure, this matches CSV row 96, Connected to Wi-Fi, under the professional fault term Can detect WiFi, but unable to connect to mobile APP. Preserve that app-level fault rather than searching for missing Wi-Fi or treating the connection problem as solved. The original rewrite is unavailable, and the recorded retrieval already includes relevant connection guidance. This is a later diagnostic stage of the same X6 incident, not an independent failure.",
        "issure_category": "terminology_mismatch",
    },
    "call_a6476bb81e7277b3dbec64447ed:user_turn_32": {
        "expected_rewrite": "Insta360 X6 cannot connect to the Insta360 app on iPhone even though the phone is connected to the camera's Wi-Fi; troubleshooting after opening iPhone Settings.",
        "correction_reason": "No original rewrite is available, so this is a proposed query from the retained conversation, not a demonstrated correction of a logged error. X6 was last named at turn 13, more than 10 transcript entries earlier; extended history may restore the model and the app-connection problem behind the user's brief confirmation. Preserve the confirmed camera Wi-Fi connection instead of turning this into a missing-Wi-Fi issue. Actual loss from the rewriter prompt cannot be verified without that log.",
        "issure_category": "out_ouf_context",
    },
    "call_a6476bb81e7277b3dbec64447ed:user_turn_44": {
        "expected_rewrite": "Insta360 X6 on iPhone: Can detect WiFi, but unable to connect to mobile APP; camera Wi-Fi is connected, but the Insta360 app still cannot connect after opening Settings and Privacy & Security.",
        "correction_reason": "CSV rows 95-96 map the earlier visible/connected-camera-Wi-Fi exchange to the exact professional fault term Can detect WiFi, but unable to connect to mobile APP. Use it to expand the ambiguous latest phrase still cannot connect my app, retaining the unresolved app-level failure and X6/iPhone context. Opening Privacy & Security does not establish that a permission was changed. The original rewrite is unavailable, and relevant guidance was retrieved despite the refusal, so a rewrite improvement is unproven. This is the final selected stage of the same X6 incident, not a separate fault type.",
        "issure_category": "terminology_mismatch",
    },
    "call_3643c8f4b7e75a67c009bcc09ac:user_turn_3": {
        "expected_rewrite": "How can I decline the repair quote and have my Insta360 X3 returned unrepaired without authorizing or paying for the repair?",
        "correction_reason": "The original reverses the direction of return: the customer wants the repair service to send the camera back, rather than to return it to Insta360. It also drops the rejected repair quote and request not to proceed with the repair. Those facts are already in the current user message, so this is intent loss rather than a 10-turn context limitation. The query requests policy information without assuming that return shipping or all charges can be waived.",
        "issure_category": "others",
    },
    "call_3643c8f4b7e75a67c009bcc09ac:user_turn_7": {
        "expected_rewrite": "Can I get my X3 camera fixed for free if it is still under warranty?",
        "correction_reason": "No substantive replacement is justified: the recorded rewrite already preserves X3 and the conditional question about free repair under warranty. Keep the same query as a control for KB scope, ranking and response-policy checks. The desired answer must explain eligibility rather than promise free repair; the observed response alone does not demonstrate a rewriting error.",
        "issure_category": "others",
    },
    "call_57d30883073e0e8a56775624334:user_turn_7": {
        "expected_rewrite": "Insta360 X5 microSD card compatibility requirements, including supported card type, speed class and capacity.",
        "correction_reason": "The original is already on topic, but the proposed wording makes technical compatibility requirements explicit instead of encouraging store recommendations limited to card capacities. Card type, speed class and capacity are dimensions to look up, not asserted specifications. This is a ranking experiment, not a confirmed rewrite fix: technical articles may be outside the logged store KB scope.",
        "issure_category": "others",
    },
    "call_57d30883073e0e8a56775624334:user_turn_13": {
        "expected_rewrite": "maximum supported microSD card capacity for X5",
        "correction_reason": "No replacement is needed: the recorded rewrite already identifies the camera, storage medium and exact maximum-capacity question. Keep it unchanged as a control; investigate KB scope, technical-document ranking and answer grounding. The evaluator's reported capacity must not be inserted into the query as an established answer.",
        "issure_category": "others",
    },
    "call_5933a98a8b1925ee7f47be4063b:user_turn_5": {
        "expected_rewrite": "Insta360 X5 purchased from Amazon with a January 2025 manufacturing date: is it old stock and should I be concerned?",
        "correction_reason": "No original rewrite is recorded, so an erroneous original cannot be established. The proposal resolves the ASR brand wording using the X5 conversation context and keeps the corrected date January 2025 rather than the abandoned January 2020 utterance. The conversation at this cutoff does not identify a battery or ask about battery extraction, so those evaluator-topic details are not inserted. Routing to human support may be the underlying issue.",
        "issure_category": "others",
    },
    "call_89bac8aaa7c1f4d637a9f2a6a91:user_turn_13": {
        "expected_rewrite": "How do I continue submitting an Insta360 Ace Pro 2 mail-in repair request after selecting the product on the Repair Service website?",
        "correction_reason": "The original already resolves the product and website, but adding mail-in repair request preserves the user's explicit preference to send the whole camera for repair instead of using a replacement lens kit. This focuses 'And then?' on the service workflow rather than product/accessory pages. That intent is available in the logged recent context, so this is not demonstrated beyond-10-turn loss. Ranking and use of previously retrieved instructions also need review.",
        "issure_category": "others",
    },
    "call_8aa8efbaa0eb9e64fc059bd2c4a:user_turn_4": {
        "expected_rewrite": "Can an extended warranty purchased from a third-party retailer be used for repair of a damaged Insta360 camera?",
        "correction_reason": "The original frames the question as repair under 'Mr. 360 warranty', blurring the user's explicit distinction between the shop's extended warranty and the manufacturer's warranty. Preserve the third-party provider and repair-eligibility question while normalizing the brand to Insta360. The relevant distinction is in the current message; neither coverage nor exclusion should be assumed.",
        "issure_category": "others",
    },
    "call_8ca923d434350f6fda924cc22ff:user_turn_7": {
        "expected_rewrite": "Which pages of the Insta360 app support landscape mode on an Android tablet, and is that device type supported?",
        "correction_reason": "The original drops Android tablet, which remains present in the logged input, and can retrieve generic landscape-mode guidance for other devices. Restore the platform and device type when resolving the follow-up about supported pages. This is omission of available context, not a demonstrated 10-turn truncation issue. The evaluator's claim that Android tablets are unsupported must be verified rather than embedded as a fact in the query.",
        "issure_category": "others",
    },
    "call_8ca923d434350f6fda924cc22ff:user_turn_21": {
        "expected_rewrite": "Insta360 app on an Android tablet cannot use landscape mode on the Playback page even after disabling Direction Lock and turning the tablet sideways.",
        "correction_reason": "The original drops Android tablet and the unsuccessful steps, although the platform and recent troubleshooting remain available in the logged input. Restoring them distinguishes a persistent device-specific problem from a first request for generic rotation instructions and may reduce repeated guidance. No confirmed tablet incompatibility or beyond-10-turn information loss is asserted.",
        "issure_category": "others",
    },
    "call_97b120465dd8a78feb3f577e283:user_turn_3": {
        "expected_rewrite": "What should I do if my Insta360 camera is stolen, and is there a way to locate or recover it?",
        "correction_reason": "No original rewrite or retained retrieval log is available, so this is a transcript-grounded proposal rather than a verified replacement. Preserve the theft/recovery intent without inventing a camera model or promising a tracking feature. The immediate handoff may result from routing before retrieval, which a rewrite change alone would not resolve.",
        "issure_category": "others",
    },
    "call_9bfd490cb5300f581a7747ac74b:user_turn_3": {
        "expected_rewrite": "What repair-service benefits and warranty coverage are available for the Insta360 X5?",
        "correction_reason": "The original broadly captures the question; the proposed wording emphasizes repair-service coverage to test ranking against lens-kit installation instructions. This is an optional clarification, not a proven original-query error. No particular purchased plan is identified before this cutoff, so the query must not assume FlexiCare, a cloud subscription or entitlement to free repair. The retrieved material and answer grounding remain possible causes.",
        "issure_category": "others",
    },
    "call_e26836d7b834515d8104744c2ad:user_turn_5": {
        "expected_rewrite": "Stolen Insta360 X4: can the camera be remotely locked or blacklisted using its serial number and proof of purchase?",
        "correction_reason": "The original rewrite is unavailable, so this is a proposed disambiguation rather than an observed wording error. In the theft context, 'lock the camera' requests remote locking or blacklisting, rather than Horizon Lock or screen lock features found in the retrieved results. The proposed query asks whether that action is possible without asserting support for it. This semantic distinction comes from the conversation, not a confirmed synonym-table mapping.",
        "issure_category": "others",
    },
}


TERMINOLOGY_MAPPING_ROWS = {
    "call_6d717fabfff7d93935428242d98:user_turn_19": [899, 900],
    "call_6d717fabfff7d93935428242d98:user_turn_27": [899, 900],
    "call_a6476bb81e7277b3dbec64447ed:user_turn_17": [95],
    "call_a6476bb81e7277b3dbec64447ed:user_turn_25": [96],
    "call_a6476bb81e7277b3dbec64447ed:user_turn_44": [95, 96],
}


def rewriting_correction(analysis):
    proposal = REWRITING_CORRECTIONS[analysis["case_id"]]
    return {"original_rewrite": analysis["original_rewrite"],
        "expected_rewrite": proposal["expected_rewrite"],
        "correction_reason": proposal["correction_reason"],
        "issure_category": proposal["issure_category"]}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raise ValueError("Unexpected message content")


def response_groups(transcript):
    groups = {}
    for index, turn in enumerate(transcript):
        if turn.get("role") != "agent" or not turn.get("words"):
            continue
        rid = str(turn.get("metadata", {}).get("response_id"))
        start, end = turn["words"][0]["start"], turn["words"][-1]["end"]
        group = groups.setdefault(rid, {"response_id": rid, "start_seconds": start,
            "end_seconds": end, "text": "", "transcript_indices": []})
        group["start_seconds"] = min(group["start_seconds"], start)
        group["end_seconds"] = max(group["end_seconds"], end)
        group["text"] += turn.get("content", "")
        group["transcript_indices"].append(index)
    return groups


def context_digest(context):
    title = re.search(r"\{H1:\s*(.*?)\}", context)
    if title:
        heading = title.group(1)
    else:
        heading = next((line for line in context.splitlines() if line.startswith("#")), context.splitlines()[0] if context else "")
    body = "\n".join(line for line in context.splitlines() if not line.startswith("{FileName:") and not line.startswith("{H1:"))
    return {"heading": heading, "excerpt": body[:650], "full_context_characters": len(context)}


def write_jsonl(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def human_review_rows(cases, replay):
    """One human-facing case per user response preceding suspect retrieval."""
    samples = {row["record_id"]: row for row in replay}
    with SYNONYMS.open(encoding="utf-8-sig", newline="") as stream:
        mappings = {index: row for index, row in enumerate(csv.DictReader(stream), 2)}
    fact_definitions = {
        "call_6d717fabfff7d93935428242d98": [
            ("camera model X4", r"\bX\s*(?:4|four)\b")],
        "call_33457cb382388d2f0599f40e7a2": [
            ("cloud service was already purchased", r"purchased.{0,40}cloud"),
            ("cloud footage is unavailable and the app requests a subscription", r"need.{0,12}subscribe")],
        "call_3db1ac9d6c2ef35abf1a8026479": [
            ("camera model Luna Ultra", r"Luna\s+Ultra")],
        "call_a6476bb81e7277b3dbec64447ed": [
            ("camera model X6", r"\bX\s*(?:6|six)\b"),
            ("phone is an iPhone", r"\biPhone\b")],
        "call_8ca923d434350f6fda924cc22ff": [
            ("device is an Android tablet", r"Android\s+tablet")],
        "call_89bac8aaa7c1f4d637a9f2a6a91": [
            ("camera model Ace Pro 2", r"Ace\s+Pro\s*2"),
            ("customer prefers sending the whole camera for repair", r"send the whole thing for repair")],
        "call_57d30883073e0e8a56775624334": [
            ("camera model X5", r"\bX\s*(?:5|five)\b")],
        "call_9bfd490cb5300f581a7747ac74b": [
            ("camera model X5", r"\bX\s*(?:5|five)\b")],
        "call_3643c8f4b7e75a67c009bcc09ac": [
            ("camera model X3", r"\bX\s*(?:3|three)\b")],
    }
    results, anchored_rows = [], {}
    for case in cases:
        sid = case["call_id"]
        for candidate in case["review_candidates"]:
            response = candidate["observed_agent_response"]
            first_index = min(response["transcript_indices"])
            preceding = [turn for turn in case["conversation"] if turn["transcript_index"] < first_index]
            user_positions = [index for index, turn in enumerate(preceding) if turn["role"] == "user"]
            if not user_positions:
                raise ValueError(f"No preceding user response for {candidate['candidate_id']}")
            before = preceding[:user_positions[-1] + 1]
            anchor = (sid, before[-1]["transcript_index"])
            sample = samples.get(candidate["rewrite_record_id"])
            if anchor in anchored_rows:
                # A repeated agent response without another user message is
                # evidence for the same retrieval test, not a new user turn.
                anchored_rows[anchor]["analysis"].setdefault("additional_agent_evidence_after_same_user_response", []).append({
                    "response_id": candidate["response_id"], "content": response["text"],
                    "original_rewrite": sample["source"]["rewritten_query"] if sample else None,
                    "rewrite_log_source": sample["retrieval_provenance"]["request_log"] if sample else None,
                    "note": "No new user response precedes this repeated agent answer; it is not a separate test input. Its later rewriter prompt may contain the first failed agent answer."})
                continue
            conversation = []
            for index, turn in enumerate(before, 1):
                item = {"turn": index, "role": turn["role"], "content": turn["content"]}
                if index == len(before):
                    item["suspected_failure_turn"] = True
                conversation.append(item)
            prompt = sample["source"]["user_content"] if sample else None
            query = sample["source"]["rewritten_query"] if sample else None
            facts = []
            for label, pattern in fact_definitions.get(sid, []):
                occurrences = [(index, turn) for index, turn in enumerate(before, 1)
                               if re.search(pattern, turn["content"], re.I)]
                if not occurrences:
                    continue
                index, occurrence = occurrences[-1]
                distance = len(before) - index
                in_prompt = bool(re.search(pattern, prompt, re.I)) if prompt is not None else None
                in_query = bool(re.search(pattern, query, re.I)) if query is not None else None
                if in_prompt is False and distance >= 10:
                    status = "earlier_fact_outside_10_turns_and_absent_from_logged_prompt"
                elif in_prompt is True and in_query is False:
                    status = "available_in_prompt_but_omitted_from_rewrite"
                elif in_prompt is True:
                    status = "retained_in_prompt_and_rewrite"
                elif in_prompt is None and distance >= 10:
                    status = "possible_window_loss_from_transcript_only_no_rewrite_log"
                else:
                    status = "not_established_as_10_turn_window_loss"
                facts.append({"information": label, "latest_prior_conversation_turn": index,
                    "quote": occurrence["content"], "intervening_transcript_entries": distance,
                    "outside_last_10_transcript_entries": distance >= 10,
                    "present_in_logged_rewriter_prompt": in_prompt,
                    "present_in_original_rewrite": in_query, "finding": status})
            confirmed_loss = [f for f in facts if f["finding"].startswith("earlier_fact_outside")]
            suspected_loss = [f for f in facts if f["finding"].startswith("possible_window_loss")]
            if confirmed_loss:
                context_status = "supported_by_earlier_dialogue_and_logged_prompt"
                context_experiment = "Prepend the actual older dialogue identified below to the exact logged input and test whether the rewrite retains those facts. Keep the existing recent prompt unchanged; finalized transcripts may contain ASR revisions. Do not insert a future answer."
            elif suspected_loss:
                context_status = "plausible_from_transcript_but_prompt_log_missing"
                context_experiment = "Candidate for extended-history testing, but obtain the original prompt/query before claiming a demonstrated 10-turn failure."
            elif any(f["finding"] == "available_in_prompt_but_omitted_from_rewrite" for f in facts):
                context_status = "relevant_fact_already_available_in_prompt"
                context_experiment = "This tests preservation of available context, not recovery from beyond 10 turns. Merely extending the window is not the primary proposed fix."
            else:
                context_status = "not_established_for_this_case"
                context_experiment = "Retained for manual review; no demonstrated beyond-10-turn information loss is asserted."
            terminology = {"status": "no_relevant_mapping_identified_in_supplied_table",
                "observed_user_wording": [], "mapping_candidates": [],
                "table_source": str(SYNONYMS.relative_to(WORKSPACE)),
                "note": "No keyword-only substitution is applied to unrelated repair, warranty, cloud-entitlement, firmware, or SD-card requests."}
            if sid == "call_6d717fabfff7d93935428242d98":
                terminology.update(status="ambiguous_mapping_candidates_require_human_confirmation",
                    observed_user_wording=[{"conversation_turn": index, "quote": turn["content"]}
                        for index, turn in enumerate(before, 1) if turn["role"] == "user" and "black line" in turn["content"].lower()],
                    note="The table maps 'black line' to Screen Flicker, other center-video descriptions to stripe/band of screen, and seam-specific lines to Black Shadow at the Seam. The evaluator suggests stitching and an earlier retrieval contains the X4 stitching/black-border article. The user does not explicitly say 'seam'; retain the symptom and consider stitching as a candidate, not a proven diagnosis.")
                mapping_rows = [452, 899, 900, 1707, 1737]
            elif sid == "call_a6476bb81e7277b3dbec64447ed":
                terminology.update(status="context_supported_mapping_candidate_original_query_unavailable",
                    observed_user_wording=[{"conversation_turn": index, "quote": turn["content"]}
                        for index, turn in enumerate(before, 1) if turn["role"] == "user" and
                        re.search(r"cannot connect|cannot,?\s+connect|yes,? it can|^\s*yes[.!]?\s*$", turn["content"], re.I)],
                    note="Use the whole earlier diagnostic exchange to distinguish detectable/connected camera Wi-Fi with app connection failure from 'Unable to Detect Wi-Fi Signal'. Some table rows are broad or inconsistent; no blind synonym replacement is used. Exact original rewrite logs are missing.")
                mapping_rows = [95, 96, 2029, 2030]
            else:
                mapping_rows = []
            for row_number in mapping_rows:
                mapping = mappings[row_number]
                terminology["mapping_candidates"].append({"csv_row": row_number,
                    "customer_description": mapping["Customer Description"],
                    "professional_term": mapping["Issue Type / Professional Fault Term"],
                    "fault_category": mapping["故障问题分类"]})
            focus = []
            if confirmed_loss or suspected_loss:
                focus.append("context_beyond_10_turns")
            if mapping_rows:
                focus.append("user_wording_to_kb_terminology")
            if not focus:
                focus.append("other_existing_issue_kept_for_manual_review")
            fields = case["evaluator_report"]["fields"]
            report = {"topic": fields["Topic（用户问的问题）"],
                "reported_failure": fields["AI答复问题点"],
                "details": fields["问题点说明（如有知识内容缺失，请备注写明）"],
                "accuracy_score": fields["AI答复准确性评分"],
                "additional_notes": fields.get("Retell", "")}
            retrieval_summary = []
            for entry in candidate["retrieval_evidence"]:
                retrieval_summary.append({"response_id": entry["response_id"],
                    "knowledgeBaseIds": entry["knowledgeBaseIds"],
                    "retrieved_article_headings": [{"rank": context["rank"], "heading": context["heading"]}
                                                   for context in entry["contexts"]],
                    "full_evidence_file": entry["source_file"], "entry_index": entry["entry_index"]})
            analysis = {"case_id": f"{sid}:user_turn_{len(before)}", "call_id": sid,
                "focus_categories": focus, "review_priority": "primary_focus" if focus[0] != "other_existing_issue_kept_for_manual_review" else "additional_case",
                "failure_turns": [t["turn"] for t in conversation if t.get("suspected_failure_turn")],
                "suspected_turn_role": "user",
                "user_response_transcript_index": before[-1]["transcript_index"],
                "rewriter_input_end_turn": len(before),
                "conversation_boundary": "Conversation stops at the user response preceding the suspect retrieval. The marked user turn is the final input turn. The resulting agent answer is analysis-only evidence and must not be added to the rewriter input.",
                "observed_agent_response": {"response_id": candidate["response_id"], "content": response["text"]},
                "why_suspicious": candidate["observation"],
                "original_rewrite": query, "original_rewriter_context": prompt,
                "rewrite_evidence": "observed_log_time_aligned_candidate" if sample else "not_present_in_supplied_export",
                "context_window_analysis": {"configured_window_turns": 10,
                    "logged_context_role_lines": len(re.findall(r"^(?:agent|customer|user):", prompt, re.M)) if prompt else None,
                    "status": context_status, "facts": facts,
                    "proposed_experiment": context_experiment,
                    "counting_note": "Distances count displayed stored user/agent transcript entries, not conversational exchanges. The exact logged prompt takes precedence because ASR revisions and transcript splits can change entry boundaries."},
                "terminology_analysis": terminology,
                "other_possible_cause": candidate["failure_category"],
                "proposed_rewrite_check": candidate["rewrite_experiment"],
                "limitations": [candidate["limitation"]] if candidate.get("limitation") else [],
                "expected_kb_article_hints": candidate["reference_article_hints"],
                "retrieval_evidence": retrieval_summary,
                "evaluator_csv_row": case["evaluator_report"]["csv_row"],
                "rewrite_log_source": sample["retrieval_provenance"]["request_log"] if sample else None,
                "causal_status": "hypothesis_for_human_review_not_confirmed_failure",
                "human_decision_reevaluate": None}
            if analysis["case_id"] in TERMINOLOGY_MAPPING_ROWS:
                selected_rows = TERMINOLOGY_MAPPING_ROWS[analysis["case_id"]]
                terms = {mappings[index]["Issue Type / Professional Fault Term"] for index in selected_rows}
                if len(terms) != 1:
                    raise ValueError(f"Conflicting selected professional terms: {analysis['case_id']}")
                selected_term = next(iter(terms))
                if selected_term not in REWRITING_CORRECTIONS[analysis["case_id"]]["expected_rewrite"]:
                    raise ValueError(f"Expected rewrite omits selected professional term: {analysis['case_id']}")
                terminology.update(selected_professional_fault_term=selected_term,
                    selected_mapping_csv_rows=selected_rows,
                    selection_status="provisional_search_term_for_human_review_not_confirmed_fault",
                    incident_id=f"{sid}:black_line" if sid == "call_6d717fabfff7d93935428242d98" else f"{sid}:app_connection",
                    independence_note="Selected user turns show stages of two underlying incidents across the five terminology samples; they are not five independent fault types.")
            if not sample:
                analysis["limitations"].extend(case["evidence_status"]["notes"])
            row = {"conversation": conversation, "evaluator_report": report,
                "rewriting_correction": rewriting_correction(analysis), "analysis": analysis}
            results.append(row)
            anchored_rows[anchor] = row
    results.sort(key=lambda r: (r["analysis"]["review_priority"] != "primary_focus",
        r["analysis"]["call_id"], r["analysis"]["failure_turns"][0]))
    return results


def main():
    with CSV.open(encoding="utf-8-sig", newline="") as stream:
        reports = {row["Session ID"].strip(): {"csv_row": i, "fields": row}
                   for i, row in enumerate(csv.DictReader(stream), 2)}
    call_ids = list(dict.fromkeys((ROOT / "call_ids.txt").read_text().split()))
    events = defaultdict(list)
    export_files = sorted((ROOT / "keyword_exports").glob("*.jsonl"))
    intervals = []
    for path in export_files:
        match = re.fullmatch(r"chunk_(.+)_to_(.+)_\d+\.jsonl", path.name)
        if match:
            intervals.append((datetime.fromisoformat(match[1]), datetime.fromisoformat(match[2])))
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                sid = (raw.get("metadata") or {}).get("call_id") or raw.get("thread_identifier")
                if sid not in call_ids or raw.get("model") != "query_rewriter":
                    continue
                messages = json.loads(raw["input"]) if isinstance(raw["input"], str) else raw["input"]
                output = json.loads(raw["output"]) if isinstance(raw["output"], str) else raw["output"]
                system = "\n".join(content_text(m["content"]) for m in messages if m["role"] == "system")
                user = "\n".join(content_text(m["content"]) for m in messages if m["role"] == "user")
                if not system or not user:
                    raise ValueError("Missing exact rewriter prompts")
                events[sid].append({"timestamp": raw["timestamp"], "system_prompt": system,
                    "user_content": user, "rewritten_query": content_text(output["content"]),
                    "latency_seconds": raw.get("latency"), "model": raw["model"],
                    "provenance": {"file": str(path.relative_to(WORKSPACE)), "line": line_number}})

    cases, replay = [], []
    for sid in call_ids:
        folder = ROOT / "calls" / sid
        call = read_json(folder / "call.json")
        transcript = read_json(folder / "transcript.json")
        groups = response_groups(transcript)
        logs = read_json(folder / "retrieval.json") if (folder / "retrieval.json").exists() else []
        log_groups = defaultdict(list)
        for index, entry in enumerate(logs):
            log_groups[str(entry["response_id"])].append((index, entry))
        start = datetime.fromtimestamp(call["startTimestamp"] / 1000, timezone.utc)
        end = datetime.fromtimestamp(call["endTimestamp"] / 1000, timezone.utc)
        selected = {item["response_id"]: item for item in ANNOTATIONS[sid]}
        dialogue = []
        for index, turn in enumerate(transcript):
            if turn.get("role") not in ("user", "agent"):
                continue
            words = turn.get("words") or []
            rid = str(turn.get("metadata", {}).get("response_id")) if turn["role"] == "agent" else None
            dialogue.append({"turn_id": f"t{len(dialogue) + 1:03d}", "transcript_index": index,
                "role": turn["role"], "content": turn.get("content", ""),
                "start_seconds": words[0]["start"] if words else None,
                "end_seconds": words[-1]["end"] if words else None,
                "response_id": rid, "heavy_issue": rid in selected if rid else False})
        rewrite_turns, prompts, candidate_events = [], {}, defaultdict(list)
        for ordinal, event in enumerate(sorted(events[sid], key=lambda e: (e["timestamp"], e["provenance"]["file"], e["provenance"]["line"])), 1):
            completion = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            seconds = (completion - start).total_seconds()
            following = sorted((g for g in groups.values() if g["start_seconds"] > seconds), key=lambda g: g["start_seconds"])
            next_response = following[0]["response_id"] if following else None
            prompt_id = hashlib.sha256(event["system_prompt"].encode()).hexdigest()[:16]
            prompts[prompt_id] = event["system_prompt"]
            record_id = hashlib.sha256(json.dumps([sid, event["timestamp"], event["provenance"]], sort_keys=True).encode()).hexdigest()
            turn = {"rewrite_index": ordinal, "record_id": record_id, "timestamp": event["timestamp"],
                "completion_seconds_from_call_start": round(seconds, 6),
                "system_prompt_id": prompt_id, "user_content": event["user_content"],
                "rewritten_query": event["rewritten_query"], "heavy_issue": False,
                "next_agent_response_candidate": next_response, "used_for_response_verified": False,
                "source": event["provenance"]}
            rewrite_turns.append(turn)
            candidate_events[next_response].append((turn, event))
        candidates = []
        for focus in ANNOTATIONS[sid]:
            rid = focus["response_id"]
            candidate_id = f"{sid}:response_{rid}"
            candidate = {"candidate_id": candidate_id, **focus,
                "observed_agent_response": groups.get(rid),
                "conversation_turn_ids": [t["turn_id"] for t in dialogue if t["response_id"] == rid],
                "rewrite_record_id": None, "selected_rewrite_status": "no_matching_rewrite_in_supplied_exports",
                "retrieval_evidence": []}
            for index, entry in log_groups[rid]:
                candidate["retrieval_evidence"].append({"source_file": str((folder / "retrieval.json").relative_to(WORKSPACE)),
                    "entry_index": index, "response_id": str(entry["response_id"]),
                    "knowledgeBaseIds": entry["kb_ids"], "logged_language": entry.get("language"),
                    "logged_kb_config": entry.get("kb_config"),
                    "contexts": [{"rank": i + 1, **context_digest(c)} for i, c in enumerate(entry.get("contexts", []))]})
            options = candidate_events[rid]
            if options:
                turn, event = options[-1]
                turn["heavy_issue"] = True
                turn["candidate_id"] = candidate_id
                candidate["rewrite_record_id"] = turn["record_id"]
                candidate["selected_rewrite_status"] = "latest_observed_completion_before_response_start_candidate"
                candidate["other_rewrite_requests_in_same_candidate_interval"] = len(options) - 1
                scope_variants = {tuple(entry["kb_ids"]) for _, entry in log_groups[rid]}
                kb_ids = list(next(iter(scope_variants))) if len(scope_variants) == 1 else None
                source = {"timestamp": event["timestamp"], "call_id": sid,
                    "customer_identifier": call["orgId"], "system_prompt": event["system_prompt"],
                    "user_content": event["user_content"], "rewritten_query": event["rewritten_query"]}
                replay.append({"conversation": dialogue, "record_id": turn["record_id"], "source": source,
                    "retrieval_settings": {"language": "en", "knowledgeBaseIds": kb_ids},
                    "review": {"candidate_id": candidate_id, "heavy_issue": True,
                        "evaluator_report": reports[sid], "analysis": focus},
                    "retrieval_provenance": {"language_source": "single_English_base_in_call_agent_configuration",
                        "kb_scope_source": "candidate_following_response_log", "historical_query_link_verified": False,
                        "request_log": event["provenance"], "response_id_candidate": rid,
                        "logged_kb_config": [entry.get("kb_config") for _, entry in log_groups[rid]],
                        "parameter_caveat": "Voice logs may contain global rather than effective node parameters; no topK/filterScore is inferred here."}})
            candidates.append(candidate)
        covered = any(left < end and right > start for left, right in intervals)
        notes = []
        if not rewrite_turns:
            notes.append("Call interval does not overlap any downloaded export file interval." if not covered else
                         "Call overlaps a downloaded interval, but no query_rewriter record for this call is present. Absence is not proof of a retrieval miss.")
        if sid == "call_5933a98a8b1925ee7f47be4063b":
            notes.append("The evaluator topic also mentions battery extraction; that request is not present in this retained conversation. Only the observed manufacturing-date concern is marked.")
        if sid == "call_9bfd490cb5300f581a7747ac74b":
            notes.append("The evaluator's cloud-benefits article and the caller's later FlexiCare plan are not interchangeable expected answers.")
        shown_rewrites = [turn for turn in rewrite_turns if turn["heavy_issue"]]
        cases.append({"conversation": dialogue, "query_rewriter_turns": shown_rewrites,
            "call_id": sid, "org_id": call["orgId"], "evaluator_report": reports[sid],
            "review_candidates": candidates, "query_rewriter_system_prompts": prompts,
            "node_transitions": [{"transcript_index": i, **t} for i, t in enumerate(transcript) if t.get("role") == "node_transition"],
            "evidence_status": {"exported_rewrite_requests_examined": len(rewrite_turns),
                "selected_rewrite_requests_shown": len(shown_rewrites),
                "unselected_rewrite_requests_omitted": len(rewrite_turns) - len(shown_rewrites),
                "retrieval_entries": len(logs),
                "call_utc_start": start.isoformat(), "call_utc_end": end.isoformat(),
                "overlaps_export_file_interval": covered, "notes": notes},
            "method": {"heavy_issue_meaning": "Selected high-impact review candidate, not a proven rewriter-caused failure.",
                "alignment": "First recorded agent response beginning after the rewrite log completion timestamp relative to stored call start. Timing association only; partial/cancelled requests may not have been used.",
                "rewrite_selection": "Only selected high-impact requests are shown, in chronological order. The latest observed completion before each selected response start is used as a candidate; intermediate partial-ASR and unselected requests remain in the original exports.",
                "replay_prompts": "Exact logged system/user text; no evaluator hints or missing historical facts are inserted.",
                "indices": "transcript_index and retrieval entry_index are zero-based; raw export line numbers are one-based."}})
    assert len({r["record_id"] for r in replay}) == len(replay)
    assert len(cases) == len(call_ids)
    write_jsonl(ROOT / "retrieval_issue_review.json", cases)
    human_rows = human_review_rows(cases, replay)
    write_jsonl(ROOT / "retrieval_issue_testset.json", human_rows)
    DATA.mkdir(parents=True, exist_ok=True)
    write_jsonl(DATA / "retrieval_issue_testset_no_analysis.json",
        [{key: value for key, value in row.items() if key != "analysis"} for row in human_rows
         if row["rewriting_correction"]["issure_category"] in {"out_ouf_context", "terminology_mismatch"}])
    print(f"Review: {len(cases)} calls, {sum(c['evidence_status']['exported_rewrite_requests_examined'] for c in cases)} rewrite requests examined, "
          f"{sum(len(c['review_candidates']) for c in cases)} selected failure points")
    print(f"Human review: {len(human_rows)} distinct user-response records, each cut before the suspect agent answer; .json suffix with JSONL contents")
    print("Calls without supplied rewrite logs:", [c["call_id"] for c in cases if not c["query_rewriter_turns"]])


if __name__ == "__main__":
    main()
