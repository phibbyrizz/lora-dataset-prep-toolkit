#!/usr/bin/env python3
"""
In-House LoRA Dataset Pipeline v0.1.4
Validated through end-to-end Windows release testing.

One command:
curated source -> robust copies -> YuNet face crops -> JoyCaption -> identity cleanup
-> hard validation -> training-ready dataset folder + ZIP.

Source is never modified. Failed builds are preserved for diagnosis.
"""

from __future__ import annotations
import argparse, csv, json, re, shutil, sys, zipfile
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
except ImportError:
    print("ERROR: Missing packages. Run: pip install -r requirements.txt", file=sys.stderr)
    raise

VERSION="0.1.4"
MODEL="fancyfeast/llama-joycaption-beta-one-hf-llava"
IMAGE_EXTS={".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".avif"}

PROMPT="""Write one concise, factual caption optimized for training an identity/character LoRA.
The subject identity will be supplied separately as the trigger token, so describe only VISUAL CONDITIONS THAT CAN CHANGE FROM IMAGE TO IMAGE.

Useful things to caption when clearly visible:
- framing and crop (close-up, head-and-shoulders, upper-body, full-body)
- pose, body orientation, head direction, gaze, and facial expression
- hairstyle, hair arrangement, and hair color when useful for that image
- clothing, footwear, jewelry, accessories, and visible makeup
- how much of the body is visible
- environment/background in generic visual terms
- lighting, camera/view angle, depth of field, and photographic style when genuinely useful

Do NOT:
- identify or guess the person's name, celebrity identity, occupation, event, exact location, brand, or relationship
- call the subject "woman", "young woman", "girl", "person", "model", "actress", or similar generic identity labels
- describe stable identity traits that the LoRA should learn from the trigger, including eye color, skin tone, facial structure, facial proportions, or body build
- transcribe, quote, identify, or describe background text, signs, logos, watermarks, banners, or branding
- use praise, personality claims, or subjective/editorial words such as beautiful, elegant, confident, glamorous, professional, attractive, poised, stylish, or sophisticated
- infer hidden body parts or clothing outside the frame
- add filler such as "the image shows", "the subject is", "simple composition", or "clear image"

Write compact natural comma-separated descriptive prose. Begin directly with the visible photographic/framing description, not with a label for the subject."""

IDENTITY_CONCEPT_PATTERNS=[
    r"\b(?:young\s+)?woman\b",r"\bgirl\b",r"\bperson\b",r"\bmodel\b",r"\bactress\b",
    r"\b(?:light|fair|pale|tan|tanned|olive|dark|medium)\s+skin\b",
    r"\bskin\s+tone\b",r"\bsmooth\s+skin(?:\s+texture)?\b",r"\bclear\s+skin\b",
    r"\b(?:blue|brown|green|hazel|gray|grey|amber)(?:[- ]brown)?\s+eyes\b",
    r"\b(?:visible\s+)?eyebrows\b", r"\b(?:straight|arched|thick|thin|defined|light(?:\s+brown)?|dark(?:\s+brown)?)\s+eyebrows\b",
    r"\beyebrows\s+(?:slightly\s+)?(?:arched|raised|furrowed|defined)\b",
    r"\b(?:visible\s+|high\s+|prominent\s+)?cheekbones\b",r"\bjawline\b",
    r"\bface\s+shape\b",r"\bfacial\s+(?:structure|proportions)\b",
    r"\b(?:background\s+|visible\s+)?text\b",r"\blogo(?:s)?\b",
    r"\bwatermark(?:s)?\b",r"\bbanner(?:s)?\b",r"\bbranding\b",
    r"\bbeautiful\b",r"\belegant\b",r"\bconfident\b",r"\bglamorous\b",
    r"\battractive\b",r"\bpoised\b",r"\bstylish\b",r"\bsophisticated\b",
]
IDENTITY_RX=[re.compile(p,re.I) for p in IDENTITY_CONCEPT_PATTERNS]

def derive_trigger(folder_name):
    name=re.sub(r"(?i)(?:[_\-\s]+(?:source|dataset))$","",folder_name).strip(" _-")
    aliases=[]
    for part in re.split(r"__+",name):
        alias=re.sub(r"[_]+"," ",part)
        alias=re.sub(r"\s+"," ",alias).strip(" ,")
        if alias: aliases.append(alias.lower())
    return ", ".join(aliases)

def parse_triggers(trig):
    aliases=[]
    for part in trig.split(","):
        alias=re.sub(r"\s+"," ",part).strip(" ,")
        if alias and alias.lower() not in {x.lower() for x in aliases}:
            aliases.append(alias)
    return aliases

def normalize_triggers(trig):
    return ", ".join(parse_triggers(trig)).lower()

def clean_generated(s):
    s=re.sub(r"^\s*(assistant|caption)\s*:\s*","",s.strip(),flags=re.I)
    s=re.sub(r"^\s*the image shows\s+","",s,flags=re.I)
    return re.sub(r"\s+"," ",s).strip(" ,.;")

def clean_identity(s,trig):
    aliases=parse_triggers(trig)
    alias_keys={x.lower() for x in aliases}
    kept=[]; removed=[]
    for concept in [x.strip() for x in s.split(",") if x.strip()]:
        c=concept.strip(" .")
        if c.lower() in alias_keys: continue
        if any(rx.search(c) for rx in IDENTITY_RX):
            removed.append(c); continue
        kept.append(c)
    return ", ".join(aliases+kept).strip(" ,"),removed

FRAMING_GROUPS = {
    "close": ("close-up", "head-and-shoulders"),
    "medium": ("medium shot", "medium-shot", "waist-up"),
    "full": ("full-body", "full body"),
}

def _has_phrase(key, phrase):
    return re.search(r"\b" + re.escape(phrase) + r"\b", key) is not None

def clean_consistency(s):
    """Conservative cleanup for duplicate and clearly contradictory caption concepts.

    This intentionally avoids guessing about subjective image content. It only resolves
    contradictions where one generated clause directly conflicts with another.
    """
    parts=[x.strip(" .") for x in s.split(",") if x.strip(" .")]
    normalized=[re.sub(r"\s+"," ",x).strip().lower() for x in parts]

    has_smile=any(_has_phrase(k,"smiling") or _has_phrase(k,"smile") for k in normalized)
    has_strapless=any(_has_phrase(k,"strapless") for k in normalized)

    # Determine the first explicit framing class. Later incompatible framing labels are
    # discarded, while descriptive visibility clauses are left untouched.
    first_frame=None
    for key in normalized:
        for group,terms in FRAMING_GROUPS.items():
            if any(_has_phrase(key,t) for t in terms):
                first_frame=group
                break
        if first_frame:
            break

    out=[]; seen=set(); seen_frame=False
    for part,key in zip(parts,normalized):
        # Exact duplicates, plus trivial variants such as "medium shot framing" after
        # "medium shot", are redundant.
        canonical=re.sub(r"\s+framing$","",key).strip()
        if canonical in seen:
            continue

        if has_smile and re.search(r"\bneutral (?:facial )?expression\b",key):
            continue
        if has_strapless and re.search(r"\b(?:dress|top|gown) covers shoulders\b",key):
            continue

        frame_group=None
        for group,terms in FRAMING_GROUPS.items():
            if any(_has_phrase(key,t) for t in terms):
                frame_group=group
                break
        if frame_group:
            if seen_frame and frame_group != first_frame:
                continue
            if frame_group == first_frame:
                if seen_frame:
                    continue
                seen_frame=True

        seen.add(canonical)
        out.append(part)
    return ", ".join(out).strip(" ,")

def semantic_flags(s):
    checks={
      "generic_subject_label":[r"\bwoman\b",r"\bgirl\b",r"\bperson\b",r"\bmodel\b",r"\bactress\b"],
      "eye_color":[r"\b(?:blue|brown|green|hazel|gray|grey|amber)(?:[- ]brown)? eyes\b"],
      "skin_identity":[r"\b(?:light|fair|pale|tan|tanned|olive|dark|medium) skin\b",r"\bskin tone\b",r"\bsmooth skin(?: texture)?\b"],
      "facial_identity":[r"\beyebrows\b",r"\bcheekbones\b",r"\bjawline\b",r"\bface shape\b",r"\bfacial structure\b",r"\bfacial proportions\b"],
      "text_or_logo":[r"\btext\b",r"\blogo(?:s)?\b",r"\bwatermark(?:s)?\b",r"\bbanner(?:s)?\b",r"\bbranding\b"],
      "subjective_style":[r"\belegant\b",r"\bconfident\b",r"\bpoised\b",r"\bstylish\b",r"\bsophisticated\b",r"\bbeautiful\b",r"\bglamorous\b",r"\battractive\b"],
    }
    lo=s.lower()
    return [label for label,pats in checks.items() if any(re.search(p,lo) for p in pats)]

def caption_ok(s,trig):
    trig=normalize_triggers(trig)
    if not s.strip(): return False,"blank"
    if not s.lower().startswith(trig.lower()+","): return False,"missing_trigger"
    if len(s)<len(trig)+12: return False,"too_short"
    flags=semantic_flags(s)
    if flags: return False,"semantic_guardrail:"+",".join(flags)
    return True,"OK"

def load_image(path):
    img=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if img is not None: return img,"opencv"
    try:
        with Image.open(path) as im:
            im=ImageOps.exif_transpose(im).convert("RGB")
            return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR),"pillow"
    except Exception:
        return None,"failed"

def crop_box(x,y,w,h,iw,ih):
    return (int(max(0,x-w*.70)),int(max(0,y-h*.80)),
            int(min(iw,x+w*1.70)),int(min(ih,y+h*2.25)))

def main():
    ap=argparse.ArgumentParser(description="One-click identity LoRA dataset builder")
    ap.add_argument("source",type=Path,help="Curated subject source folder")
    ap.add_argument("--trigger",default=None,help="Override trigger; default derives from folder name")
    ap.add_argument("--model",type=Path,default=None,help="YuNet ONNX path; default searches beside this script")
    ap.add_argument("--output",type=Path,default=None,help="Output root; default source parent/_pipeline_output")
    ap.add_argument("--min-confidence",type=float,default=.90)
    ap.add_argument("--min-face-px",type=int,default=80)
    ap.add_argument("--max-new-tokens",type=int,default=160)
    a=ap.parse_args()

    source=a.source.resolve()
    if not source.is_dir(): sys.exit("Source folder not found: "+str(source))

    trig=normalize_triggers(a.trigger or derive_trigger(source.name))
    if not trig: sys.exit("Could not derive trigger. Use --trigger.")

    script_dir=Path(__file__).resolve().parent
    model_name="face_detection_yunet_2023mar.onnx"
    model_candidates=[script_dir/model_name,script_dir.parent/model_name,Path.cwd()/model_name]
    yunet=(a.model.resolve() if a.model else next((p for p in model_candidates if p.is_file()),model_candidates[0]))
    if not yunet.is_file():
        searched="\n".join("  - "+str(p) for p in model_candidates)
        sys.exit("YuNet model not found. Searched:\n"+searched+"\nPlace "+model_name+" beside this script (or one folder above it), or use --model.")

    if not torch.cuda.is_available(): sys.exit("CUDA GPU not available; JoyCaption requires CUDA in this pipeline.")
    if not hasattr(cv2,"FaceDetectorYN"): sys.exit("OpenCV FaceDetectorYN unavailable. Upgrade opencv-python.")

    outroot=(a.output.resolve() if a.output else source.parent/"_pipeline_output")
    outroot.mkdir(parents=True,exist_ok=True)
    subject=source.name
    final_dir=outroot/f"{subject}_dataset"
    temp_dir=outroot/f"{subject}_BUILDING"
    reports=outroot/"_pipeline_reports"/subject
    reports.mkdir(parents=True,exist_ok=True)
    final_zip=outroot/f"{subject}_TRAINING_READY.zip"

    if final_dir.exists() or temp_dir.exists() or final_zip.exists():
        sys.exit("Existing output detected. Refusing to overwrite.\n"
                 f"Dataset: {final_dir}\nBuild workspace: {temp_dir}\nZIP: {final_zip}")

    images=sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images: sys.exit("No source images found.")
    stems={}
    for p in images:
        stems.setdefault(p.stem.casefold(),[]).append(p.name)
    duplicate_stems=[names for names in stems.values() if len(names)>1]
    if duplicate_stems:
        details="\n".join("  - "+", ".join(names) for names in duplicate_stems)
        sys.exit("Source contains image files with duplicate base names. Each image needs a unique base name "
                 "because its caption uses the same .txt stem. Rename these before retrying:\n"+details)

    print("\n=== IN-HOUSE LoRA DATASET PIPELINE v0.1.4 ===")
    print("Source :",source)
    print("Trigger:",trig)
    print("Images :",len(images))
    print("GPU    :",torch.cuda.get_device_name(0))
    print("Source remains read-only.\n")

    temp_dir.mkdir(parents=True)
    detector=cv2.FaceDetectorYN.create(str(yunet),"",(320,320),a.min_confidence,.3,5000)
    crop_rows=[]; source_failures=[]

    print("=== STAGE 1: ORIGINALS + FACE AUGMENTATION ===")
    for i,src in enumerate(images,1):
        img,loader=load_image(src)
        if img is None:
            source_failures.append(src.name)
            crop_rows.append([src.name,"SOURCE","FAILED","","unreadable"])
            print(f"[{i}/{len(images)}] FAILED {src.name}")
            continue
        shutil.copy2(src,temp_dir/src.name)
        crop_rows.append([src.name,"ORIGINAL","OK","",loader])
        h,w=img.shape[:2]
        detector.setInputSize((w,h))
        _,faces=detector.detect(img)
        faces=[] if faces is None else sorted(faces,key=lambda f:float(f[-1]),reverse=True)
        confident=[f for f in faces if float(f[-1])>=a.min_confidence]
        cropmade=False
        if confident:
            f=confident[0]; x,y,fw,fh=map(float,f[:4])
            if min(fw,fh)>=a.min_face_px:
                x1,y1,x2,y2=crop_box(x,y,fw,fh,w,h)
                if ((x2-x1)*(y2-y1))/(w*h)<=.82:
                    cp=temp_dir/f"{src.stem}_facecrop.png"
                    if cv2.imwrite(str(cp),img[y1:y2,x1:x2]):
                        note="multiple_faces_review" if len(confident)>1 else "generated"
                        crop_rows.append([cp.name,"FACECROP","OK",src.name,note]); cropmade=True
        print(f"[{i}/{len(images)}] {src.name}"+(" + crop" if cropmade else ""))

    with (reports/"build_report.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["file","type","status","derived_from","note"]); w.writerows(crop_rows)

    if source_failures:
        (reports/"FAILED.txt").write_text("Unreadable source images:\n"+"\n".join(source_failures),encoding="utf-8")
        sys.exit(f"BUILD BLOCKED: {len(source_failures)} unreadable source image(s). Failed workspace preserved: {temp_dir}")

    dataset_imgs=sorted(p for p in temp_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    originals=sum(1 for r in crop_rows if r[1]=="ORIGINAL" and r[2]=="OK")
    crops=sum(1 for r in crop_rows if r[1]=="FACECROP" and r[2]=="OK")
    print(f"\nStage 1 complete: {originals} originals + {crops} face crops = {len(dataset_imgs)} images")

    print("\n=== STAGE 2: JOYCAPTION + IDENTITY CLEANUP ===")
    try:
        proc=AutoProcessor.from_pretrained(MODEL)
        q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["vision_tower","multi_modal_projector"])
        model=LlavaForConditionalGeneration.from_pretrained(
            MODEL,torch_dtype="auto",device_map=0,quantization_config=q)
        model.eval()
    except torch.cuda.OutOfMemoryError as e:
        shutil.rmtree(temp_dir,ignore_errors=True)
        sys.exit("JoyCaption could not load because GPU memory was exhausted. "
                 "Close other GPU-heavy applications and retry. GPU: "+torch.cuda.get_device_name(0)+"\n"+repr(e))
    except Exception as e:
        shutil.rmtree(temp_dir,ignore_errors=True)
        sys.exit("JoyCaption model loading failed. Run the launcher preflight again and check "
                 "the generated error. Details: "+repr(e))
    convo=[{"role":"system","content":"You are a helpful image captioner."},
           {"role":"user","content":PROMPT}]
    convo_string=proc.apply_chat_template(convo,tokenize=False,add_generation_prompt=True)
    if not isinstance(convo_string,str): sys.exit("JoyCaption chat template failure.")

    validation=[]; genfail=[]; cleanup_log=[]
    for i,p in enumerate(dataset_imgs,1):
        print(f"[{i}/{len(dataset_imgs)}] {p.name}",flush=True)
        txt=p.with_suffix(".txt")
        try:
            with Image.open(p) as im: image=im.convert("RGB")
            inputs=proc(text=[convo_string],images=[image],return_tensors="pt").to("cuda")
            inputs["pixel_values"]=inputs["pixel_values"].to(torch.bfloat16)
            with torch.inference_mode():
                out=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,
                                   suppress_tokens=None,use_cache=True)[0]
            out=out[inputs["input_ids"].shape[1]:]
            body=clean_generated(proc.tokenizer.decode(out,skip_special_tokens=True,
                    clean_up_tokenization_spaces=False))
            if not body: raise RuntimeError("JoyCaption returned empty caption")
            cap,removed=clean_identity(f"{trig}, {body}",trig)
            cap=clean_consistency(cap)
            ok,reason=caption_ok(cap,trig)
            if not ok:
                validation.append([p.name,reason]); cleanup_log.append([p.name," | ".join(removed)])
                rejected_dir=reports/"rejected_captions"
                rejected_dir.mkdir(parents=True,exist_ok=True)
                (rejected_dir/(p.stem+".txt")).write_text(cap+"\n",encoding="utf-8")
                continue
            txt.write_text(cap+"\n",encoding="utf-8")
            validation.append([p.name,"OK"]); cleanup_log.append([p.name," | ".join(removed)])
            print("  ->",cap[:200],flush=True)
        except Exception as e:
            genfail.append([p.name,repr(e)]); validation.append([p.name,"GENERATION_FAILED:"+type(e).__name__])
            print("  !!",repr(e),flush=True)

    # Hard final gate: every image has one meaningful current caption.
    final_fail=[]
    for p in dataset_imgs:
        txt=p.with_suffix(".txt")
        if not txt.is_file():
            final_fail.append([p.name,"missing_caption"]); continue
        cap=txt.read_text(encoding="utf-8",errors="replace").strip()
        ok,reason=caption_ok(cap,trig)
        if not ok: final_fail.append([p.name,reason])

    with (reports/"caption_validation.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["image","status"]); w.writerows(validation)
    with (reports/"cleanup_log.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["image","removed_concepts"]); w.writerows(cleanup_log)

    rejected_rows=[]
    rejected_dir=reports/"rejected_captions"
    if rejected_dir.is_dir():
        status_by_image={r[0]:r[1] for r in validation}
        for rp in sorted(rejected_dir.glob("*.txt")):
            image_name=next((x.name for x in dataset_imgs if x.stem==rp.stem),rp.stem)
            rejected_rows.append([image_name,status_by_image.get(image_name,"UNKNOWN"),
                                  rp.read_text(encoding="utf-8",errors="replace").strip()])
    with (reports/"rejected_captions.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["image","failure_reason","post_cleanup_caption"]); w.writerows(rejected_rows)

    manifest={
      "pipeline_version":VERSION,"caption_calibration":"JoyCaption v0.6.1",
      "source_folder":str(source),"subject_folder":subject,"trigger":trig,
      "timestamp":datetime.now().isoformat(timespec="seconds"),
      "gpu":torch.cuda.get_device_name(0),"source_images":len(images),
      "originals":originals,"face_crops":crops,"dataset_images":len(dataset_imgs),
      "valid_captions":len(dataset_imgs)-len(final_fail),"invalid_captions":len(final_fail),
      "generation_failures":genfail,"final_gate_failures":final_fail,
      "training_ready":not final_fail and not genfail
    }
    (reports/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    print("\n=== FINAL VALIDATION ===")
    print("Trigger             :",trig)
    print("Originals           :",originals)
    print("Face crops          :",crops)
    print("Dataset images      :",len(dataset_imgs))
    print("Valid captions      :",len(dataset_imgs)-len(final_fail))
    print("Invalid captions    :",len(final_fail))
    print("Generation failures :",len(genfail))

    if final_fail or genfail:
        print("\nTRAINING ZIP BLOCKED.")
        print("Failed workspace preserved:",temp_dir)
        print("Reports:",reports)
        sys.exit(2)

    # Successful build: promote workspace and create clean training ZIP containing pairs only.
    temp_dir.rename(final_dir)
    with zipfile.ZipFile(final_zip,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(final_dir.iterdir()):
            if p.is_file() and (p.suffix.lower() in IMAGE_EXTS or p.suffix.lower()==".txt"):
                z.write(p,p.name)
    try:
        with zipfile.ZipFile(final_zip,"r") as z:
            bad=z.testzip()
            if bad:
                raise RuntimeError("ZIP integrity check failed at "+bad)
    except Exception:
        final_zip.unlink(missing_ok=True)
        raise

    print("\nTRAINING READY")
    print("Dataset:",final_dir)
    print("ZIP    :",final_zip)
    print("Reports:",reports)

if __name__=="__main__":
    main()
