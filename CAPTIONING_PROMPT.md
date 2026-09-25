# Captioning prompt

The pipeline currently sends the following custom instruction to JoyCaption for every image. The subject trigger is added separately by the pipeline.

```text
Write one concise, factual caption optimized for training an identity/character LoRA.
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

Write compact natural comma-separated descriptive prose. Begin directly with the visible photographic/framing description, not with a label for the subject.
```
