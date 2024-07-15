#! /usr/bin/env python3


from typing import List

import torch
import torch.nn.functional as F
from _svd_perplexity_gpt_model import GPTModel, GPTTokenizer

GPT_PARAM_PATH = "/u/qxc4fh/zeyu_workspace/gpt_params.pkl"
GPT_TOKENIZER_KERNEL_PATH = "/u/qxc4fh/zeyu_workspace/gpt_tokenizer_kernel.pkl"
MODEL_DEVICE = "cuda:0"


INF_TARGET_TENSOR = None


def pad_batch(batch, pad_id, max_len):
    context_lengths = []
    max_context_length = max([len(tokens) for tokens in batch])
    for tokens in batch:
        context_length = len(tokens)
        if context_length < max_context_length + max_len:
            tokens.extend([pad_id] * (max_context_length + max_len - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def tokenize_batch(tokenizer, sentences, max_len, add_BOS=True):
    if add_BOS:
        context_tokens = [[tokenizer.eos_id] + tokenizer.text_to_ids(s) for s in sentences]
    else:
        context_tokens = [tokenizer.text_to_ids(s) for s in sentences]
    context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len)
    context_tokens_tensor = torch.LongTensor(context_tokens)
    context_length_tensor = torch.LongTensor(context_lengths)
    return context_tokens_tensor, context_length_tensor


def get_ltor_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, position_ids


def get_batch(context_tokens):
    """Generate batch from context tokens."""
    # Move to GPU.
    tokens = context_tokens.contiguous()  # .cuda()
    # Get the attention mask and postition ids.
    attention_mask, position_ids = get_ltor_masks_and_position_ids(tokens)

    return tokens, attention_mask, position_ids


def repetition_penalty(logits, repetition_penalty, used_tokens):
    """Implement the repetition penalty, check paper
    https://arxiv.org/pdf/1909.05858.pdf
    """
    if used_tokens is not None and repetition_penalty != 1.0:
        logits_update = torch.gather(logits, 1, used_tokens)
        logits = torch.scatter(logits, 1, used_tokens, logits_update / repetition_penalty)
    return logits


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """This function has been mostly taken from huggingface conversational
    ai code at
        https://medium.com/huggingface/how-to-build-a-state-of-the-art-
             conversational-ai-with-transfer-learning-2d818ac26313"""

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(
    model,
    tokenizer,
    context_tokens,
    context_lengths,
    attention_mask,
    position_ids,
    tokens_to_generate,
    temperature_value=1.0,
    repetition_penalty_value=1.2,
    top_k_value=0,
    top_p_value=0.9,
):
    micro_batch_size = context_tokens.shape[0]
    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()

        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = tokenizer.eos_id
        counter = 0

        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size], device=context_tokens.device).byte()
        tokens = context_tokens
        output_logits = None
        all_generated_indices = None  # used to track all generated indices
        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + context_lengths.max().item()

        if maxlen > model.encoder_seq_length + 1:
            maxlen = model.encoder_seq_length + 1

        while context_length < maxlen:
            # types2use = None
            if counter == 0:
                # Allocate memory for the entire context.
                set_inference_key_value_memory = True
                tokens2use = tokens[:, :context_length]
                positions2use = position_ids[:, :context_length]
                # not using type2use. uncomment it if it is used
                # if type_ids is not None:
                #     types2use = type_ids[:, :context_length]
            else:
                # Set this to false so the memory is not reallocated.
                set_inference_key_value_memory = False
                tokens2use = tokens[:, context_length - 1].view(batch_size, -1)
                positions2use = position_ids[:, context_length - 1].view(batch_size, -1)
                # not using type2use. uncomment it if it is used
                # if type_ids is not None:
                #     types2use = type_ids[:, context_length - 1].view(batch_size, -1)

            attention_mask_repeat = torch.concat([attention_mask for _ in range(micro_batch_size)])
            setkey_value_array = torch.tensor([set_inference_key_value_memory] * micro_batch_size)
            len_array = torch.tensor([maxlen] * micro_batch_size)

            # Only prompt learning models will have a prompt table, and require task ids
            batch = [tokens2use, attention_mask_repeat, positions2use, setkey_value_array, len_array]

            output = model(batch, INF_TARGET_TENSOR)

            assert output is not None
            output = output.float()
            logits = output[:, -1].view(batch_size, -1).contiguous()

            # make sure it won't sample outside the vocab_size range
            logits[:, tokenizer.vocab_size :] = -float("Inf")

            logits = logits.float()
            logits /= temperature_value
            # handle repetition penality
            logits = repetition_penalty(logits, repetition_penalty_value, all_generated_indices)
            logits = top_k_logits(logits, top_k=top_k_value, top_p=top_p_value)
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            started = context_lengths <= context_length

            # Clamp the predicted out of vocabulary tokens
            prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
            new_tokens = switch(tokens[:, context_length].view(-1), prev, started)

            # Replace sampled tokens w/ done token if EOD has already been sampled
            new_tokens = switch(new_tokens, eod_id, is_done)

            # Insert either new predicted or next prompt token
            tokens[:, context_length] = new_tokens

            if output_logits is None:
                output = F.log_softmax(output[:, :context_length, :], 2)
                indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                output_logits = torch.gather(output, 2, indices).squeeze(2)
                all_generated_indices = indices[:, :, 0]
            else:
                output = F.log_softmax(output, 2)
                indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                new_output_logits = torch.gather(output, 2, indices).squeeze(2)

                # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                output_logits = torch.cat([output_logits, new_output_logits], 1)
                all_generated_indices = torch.cat([all_generated_indices, indices[:, :, 0]], 1)

            done_token = (prev == eod_id).byte() & started.byte()
            is_done = is_done | done_token

            done = torch.all(is_done)

            yield tokens, output_logits

            context_length += 1
            counter += 1
            if done:
                break


def gpt_generate(inputs: List[str], max_gen_len):
    tokenizer = GPTTokenizer(GPT_TOKENIZER_KERNEL_PATH)
    context_tokens_tensor, context_length_tensor = tokenize_batch(tokenizer, inputs, max_gen_len)
    context_tokens_tensor = context_tokens_tensor.to(torch.device(MODEL_DEVICE))
    context_length_tensor = context_length_tensor.to(torch.device(MODEL_DEVICE))

    target_tensor = context_tokens_tensor[:, 1:2049]
    print(target_tensor.shape)
    target_tensor = target_tensor[0, :]
    print(target_tensor.shape)
    global INF_TARGET_TENSOR
    INF_TARGET_TENSOR = target_tensor
    context_tokens_tensor = context_tokens_tensor[:, :2049]
    context_length_tensor = torch.tensor([2048], dtype=torch.int64, device=context_length_tensor.device)

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)

    model = GPTModel(GPT_PARAM_PATH, 16, MODEL_DEVICE)
    batch_token_iterator = sample_sequence_batch(
        model,
        tokenizer,
        context_tokens_tensor,
        context_length_tensor,
        attention_mask,
        position_ids,
        max_gen_len,
    )

    for tokens, output_logits in batch_token_iterator:
        context_length += 1

    decode_tokens = tokens[:, :context_length]
    decode_tokens = decode_tokens.cpu().numpy().tolist()
    resp_sentences = []
    resp_sentences_seg = []
    for decode_token in decode_tokens:
        sentence = tokenizer.ids_to_text(decode_token)
        resp_sentences.append(sentence)

        words = []
        for token in decode_token:
            # Skip any soft prompt pseudo tokens
            if token not in tokenizer.tokenizer.decoder:
                print(1)
                continue
            word = tokenizer.tokenizer.decoder[token]
            word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode("utf-8", errors="replace")
            words.append(word)
        resp_sentences_seg.append(words)

    return resp_sentences, resp_sentences_seg


if __name__ == "__main__":
    inputs = [
        "It is done, and submitted. You can play “Survival of the Tastiest” on Android, and on the web. Playing on the web works, but you have to simulate multi-touch for table moving and that can be a bit confusing. There’s a lot I’d like to talk about. I’ll go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise – something with lots of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident I could fit any theme around it. In the end, the problem with a theme like “Evolution” in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it’s not evolution anymore – it’s the equivalent of intelligent design, the fable invented by creationists to combat the very idea of evolution. Being agnostic and a Pastafarian, that’s not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn’t want to create an “intelligent design” simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I’d say the only real solution was through the use of artificial selection, somehow. So far, I haven’t seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn’t think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the “Mechanical Underdogs” signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist – by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your “base”. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by “ordering” them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying costs, which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn’t like other people’s pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, until one of them die (think Battlefield’s standard “Conquest” mode). You get points every second for every plate you own. Over time, the concept also evolved to use an Italian bistro as its main scenario. Carlos, Carlos’ Bistro’s founder and owner Setup No major changes were made from my work setup. I used FDT and Starling creating an Adobe AIR (ActionScript) project, all tools or frameworks I already had some knowledge with. One big change for me was that I livestreamed my work through a twitch.tv account. This was a new thing for me. As recommended by Roushey, I used a program called XSplit and I got to say, it is pretty amazing. It made the livestream pretty effortless and the features are awesome, even for the free version. It was great to have some of my friends watch me, and then interact with them and random people through chat. It was also good knowing that I was also recording a local version of the files, so I could make a timelapse video later. Knowing the video was being recorded also made me a lot more self-conscious about my computer use, as if someone was watching over my shoulder. It made me realize that sometimes I spend too much time in seemingly inane tasks (I ended up wasting the longest time just to get some text alignment the way I wanted – it’ll probably drive someone crazy if they watch it) and that I do way too many typos where writing code. I pretty much spend half of the time writing a line and the other half fixing the crazy characters in it. My own stream was probably boring to watch since I was coding for the most time. But livestreaming is one of the cool things to do as a spectator too. It was great seeing other people working – I had a few tabs opened on my second monitor all the time. It’s actually a bit sad, because if I could, I could have spent the whole weekend just watching other people working! But I had to do my own work, so I’d only do it once in a while, when resting for a bit. Design Although I wanted some simple, low-fi, high-contrast kind of design, I ended up going with somewhat realistic (vector) art. I think it worked very well, fitting the mood of the game, but I also went overboard. For example: to know the state of a plate (who owns it, who’s conquering it and how much time they have left before conquering it, which pasta units are in the queue, etc), you have to look at the plate’s bill. The problem I realized when doing some tests is that people never look at the bill! They think it’s some kind of prop, so they never actually read its details. Plus, if you’re zoomed out too much, you can’t actually read it, so it’s hard to know what’s going on with the game until you zoom in to the area of a specific plate. One other solution that didn’t turn out to be as perfect as I thought was how to indicate who a plate base belongs to. In the game, that’s indicated by the plate’s decoration – its color denotes the team owner. But it’s something that fits so well into the design that people never realized it, until they were told about it. In the end, the idea of going with a full physical metaphor is one that should be done with care. Things that are very important risk becoming background noise, unless the player knows its importance. Originally, I wanted to avoid any kind of heads-up display in my game. In the end, I ended up adding it at the bottom to indicate your credits and bases owned, as well as the hideous out-of-place-and-still-not-obvious “Call Waiter” button. But in hindsight, I should have gone with a simple HUD from the start, especially one that indicated each team’s colors and general state of the game without the need for zooming in and out. Development Development went fast. But not fast enough. Even though I worked around 32+ hours for this Ludum Dare, the biggest problem I had to face in the end was overscoping. I had too much planned, and couldn’t get it all done. Content-wise, I had several kinds of pasta planned (Wikipedia is just amazing in that regard), split into several different groups, from small Pastina to huge Pasta al forno. But because of time constraints, I ended up scratching most of them, and ended up with 5 different types of very small pasta – barely something to start when talking about the evolution of Pasta. Pastas used in the game. Unfortunately, the macs where never used Which is one of the saddest things about the project, really. It had the framework and the features to allow an endless number of elements in there, but I just didn’t have time to draw the rest of the assets needed (something I loved to do, by the way). Other non-obvious features had to be dropped, too. For example, when ordering some pasta, you were supposed to select what kind of sauce you’d like with your pasta, each with different attributes. Bolognese, for example, is very strong, but inaccurate; Pesto is very accurate and has great range, but it’s weaker; and my favorite, Vodka, would triggers 10% loss of speed on the pasta hit by it. The code for that is mostly in there. But in the end, I didn’t have time to implement the sauce selection interface; all pasta ended up using bolognese sauce. To-do list: lots of things were not done Actual programming also took a toll in the development time. Having been programming for a while, I like to believe I got to a point where I know how to make things right, but at the expense of forgetting how to do things wrong in a seemingly good way. What I mean is that I had to take a lot of shortcuts in my code to save time (e.g. a lot of singletons references for cross-communication rather than events or observers, all-encompassing check loops, not fast enough) that left a very sour taste in my mouth. While I know I used to do those a few years ago and survive, I almost cannot accept the state my code is in right now. At the same time, I do know it was the right thing to do given the timeframe. One small thing that had some impact was using a somewhat new platform for me. That’s Starling, the accelerated graphics framework I used in Flash. I had tested it before and I knew how to use it well – the API is very similar to Flash itself. However, there were some small details that had some impact during development, making me feel somewhat uneasy the whole time I was writing the game. It was, again, the right thing to do, but I should have used Starling more deeply before (which is the conundrum: I used it for Ludum Dare just so I could learn more about it). Argument and user experience One final aspect of the game that I learned is that making the game obvious for your players goes a long way into making it fun. If you have to spend the longest time explaining things, your game is doing something wrong. And that’s exactly the problem Survival of the Tastiest ultimately faced. It’s very hard for people to understand what’s going on with the game, why, and how. I did have some introductory text at the beginning, but that was a last-minute thing. More importantly, I should have had a better interface or simplified the whole concept so it would be easier for people to understand. That doesn’t mean the game itself should be simple. It just means that the experience and interface should be approachable and understandable. Conclusion I’m extremely happy with what I’ve done and, especially given that this was my first Ludum Dare. However, I feel like I’ve learned a lot of what not to do. The biggest problem is overscoping. Like Eric Decker said, the biggest lesson we can learn with this is probably with scoping – deciding what to do beforehand in a way you can complete it without having to rush and do something half-assed. I’m sure I will do more Ludum Dares in the future. But if there are any lessons I can take of it, they are to make it simple, to use frameworks and platforms you already have some absolute experience with (otherwise you’ll spend too much time trying to solve easy questions), and to scope for a game that you can complete in one day only (that way, you can actually take two days and make it cool). This entry was posted on Monday, August 27th, 2012 at 10:54 am and is filed under LD #24. You can follow any responses to this entry through the RSS 2.0 feed. You can skip to the end and leave a response. Pinging is currently not allowed. 3 Responses to ““Survival of the Tastiest” Post-mortem” darn it , knowing that I missed your livestream makes me a sad panda ;( but more to the point, the game is … well for a startup its original to say the least ;D it has some really neat ideas and more importantly its designed arround touch screens whitch by the looks of the submission is something rare ;o or that could be just me and my short memory -_-! awesum game, love et <3"
    ]
    # inputs = [
    #     "Your pasta doesn’t like other people’s pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, until one of them die (think Battlefield’s standard “Conquest” mode). You get points every second for every plate you own. Over time, the concept also evolved to use an Italian bistro as its main scenario. Carlos, Carlos’ Bistro’s founder and owner Setup No major changes were made from my work setup. I used FDT and Starling creating an Adobe AIR (ActionScript) project, all tools or frameworks I already had some knowledge with. One big change for me was that I livestreamed my work through a twitch.tv account. This was a new thing for me. As recommended by Roushey, I used a program called XSplit and I got to say, it is pretty amazing. It made the livestream pretty effortless and the features are awesome, even for the free version. It was great to have some of my friends watch me, and then interact with them and random people through chat. It was also good knowing that I was also recording a local version of the files, so I could make a timelapse video later. Knowing the video was being recorded also made me a lot more self-conscious about my computer use, as if someone was watching over my shoulder. It made me realize that sometimes I spend too much time in seemingly inane tasks (I ended up wasting the longest time just to get some text alignment the way I wanted – it’ll probably drive someone crazy if they watch it) and that I do way too many typos where writing code. I pretty much spend half of the time writing a line and the other half fixing the crazy characters in it. My own stream was probably boring to watch since I was coding for the most time. But livestreaming is one of the cool things to do as a spectator too. It was great seeing other people working – I had a few tabs opened on my second monitor all the time. It’s actually a bit sad, because if I could, I could have spent the whole weekend just watching other people working! But I had to do my own work, so I’d only do it once in a while, when resting for a bit. Design Although I wanted some simple, low-fi, high-contrast kind of design, I ended up going with somewhat realistic (vector) art. I think it worked very well, fitting the mood of the game, but I also went overboard. For example: to know the state of a plate (who owns it, who’s conquering it and how much time they have left before conquering it, which pasta units are in the queue, etc), you have to look at the plate’s bill. The problem I realized when doing some tests is that people never look at the bill! They think it’s some kind of prop, so they never actually read its details. Plus, if you’re zoomed out too much, you can’t actually read it, so it’s hard to know what’s going on with the game until you zoom in to the area of a specific plate. One other solution that didn’t turn out to be as perfect as I thought was how to indicate who a plate base belongs to. In the game, that’s indicated by the plate’s decoration – its color denotes the team owner. But it’s something that fits so well into the design that people never realized it, until they were told about it. In the end, the idea of going with a full physical metaphor is one that should be done with care. Things that are very important risk becoming background noise, unless the player knows its importance. Originally, I wanted to avoid any kind of heads-up display in my game. In the end, I ended up adding it at the bottom to indicate your credits and bases owned, as well as the hideous out-of-place-and-still-not-obvious “Call Waiter” button. But in hindsight, I should have gone with a simple HUD from the start, especially one that indicated each team’s colors and general state of the game without the need for zooming in and out. Development Development went fast. But not fast enough. Even though I worked around 32+ hours for this Ludum Dare, the biggest problem I had to face in the end was overscoping. I had too much planned, and couldn’t get it all done. Content-wise, I had several kinds of pasta planned (Wikipedia is just amazing in that regard), split into several different groups, from small Pastina to huge Pasta al forno. But because of time constraints, I ended up scratching most of them, and ended up with 5 different types of very small pasta – barely something to start when talking about the evolution of Pasta. Pastas used in the game. Unfortunately, the macs where never used Which is one of the saddest things about the project, really. It had the framework and the features to allow an endless number of elements in there, but I just didn’t have time to draw the rest of the assets needed (something I loved to do, by the way). Other non-obvious features had to be dropped, too. For example, when ordering some pasta, you were supposed to select what kind of sauce you’d like with your pasta, each with different attributes. Bolognese, for example, is very strong, but inaccurate; Pesto is very accurate and has great range, but it’s weaker; and my favorite, Vodka, would triggers 10% loss of speed on the pasta hit by it. The code for that is mostly in there. But in the end, I didn’t have time to implement the sauce selection interface; all pasta ended up using bolognese sauce. To-do list: lots of things were not done Actual programming also took a toll in the development time. Having been programming for a while, I like to believe I got to a point where I know how to make things right, but at the expense of forgetting how to do things wrong in a seemingly good way. What I mean is that I had to take a lot of shortcuts in my code to save time (e.g. a lot of singletons references for cross-communication rather than events or observers, all-encompassing check loops, not fast enough) that left a very sour taste in my mouth. While I know I used to do those a few years ago and survive, I almost cannot accept the state my code is in right now. At the same time, I do know it was the right thing to do given the timeframe. One small thing that had some impact was using a somewhat new platform for me. That’s Starling, the accelerated graphics framework I used in Flash. I had tested it before and I knew how to use it well – the API is very similar to Flash itself. However, there were some small details that had some impact during development, making me feel somewhat uneasy the whole time I was writing the game. It was, again, the right thing to do, but I should have used Starling more deeply before (which is the conundrum: I used it for Ludum Dare just so I could learn more about it). Argument and user experience One final aspect of the game that I learned is that making the game obvious for your players goes a long way into making it fun. If you have to spend the longest time explaining things, your game is doing something wrong. And that’s exactly the problem Survival of the Tastiest ultimately faced. It’s very hard for people to understand what’s going on with the game, why, and how. I did have some introductory text at the beginning, but that was a last-minute thing. More importantly, I should have had a better interface or simplified the whole concept so it would be easier for people to understand. That doesn’t mean the game itself should be simple. It just means that the experience and interface should be approachable and understandable. Conclusion I’m extremely happy with what I’ve done and, especially given that this was my first Ludum Dare. However, I feel like I’ve learned a lot of what not to do. The biggest problem is overscoping. Like Eric Decker said, the biggest lesson we can learn with this is probably with scoping – deciding what to do beforehand in a way you can complete it without having to rush and do something half-assed. I’m sure I will do more Ludum Dares in the future. But if there are any lessons I can take of it, they are to make it simple, to use frameworks and platforms you already have some absolute experience with (otherwise you’ll spend too much time trying to solve easy questions), and to scope for a game that you can complete in one day only (that way, you can actually take two days and make it cool). This entry was posted on Monday, August 27th, 2012 at 10:54 am and is filed under LD #24. You can follow any responses to this entry through the RSS 2.0 feed. You can skip to the end and leave a response. Pinging is currently not allowed. 3 Responses to ““Survival of the Tastiest” Post-mortem” darn it , knowing that I missed your livestream makes me a sad panda ;( but more to the point, the game is … well for a startup its original to say the least ;D it has some really neat ideas and more importantly its designed arround touch screens whitch by the looks of the submission is something rare ;o or that could be just me and my short memory -_-! awesum game, love et <3"
    # ]
    # inputs = [
    #     "Bolognese, for example, is very strong, but inaccurate; Pesto is very accurate and has great range, but it’s weaker; and my favorite, Vodka, would triggers 10% loss of speed on the pasta hit by it. The code for that is mostly in there. But in the end, I didn’t have time to implement the sauce selection interface; all pasta ended up using bolognese sauce. To-do list: lots of things were not done Actual programming also took a toll in the development time. Having been programming for a while, I like to believe I got to a point where I know how to make things right, but at the expense of forgetting how to do things wrong in a seemingly good way. What I mean is that I had to take a lot of shortcuts in my code to save time (e.g. a lot of singletons references for cross-communication rather than events or observers, all-encompassing check loops, not fast enough) that left a very sour taste in my mouth. While I know I used to do those a few years ago and survive, I almost cannot accept the state my code is in right now. At the same time, I do know it was the right thing to do given the timeframe. One small thing that had some impact was using a somewhat new platform for me. That’s Starling, the accelerated graphics framework I used in Flash. I had tested it before and I knew how to use it well – the API is very similar to Flash itself. However, there were some small details that had some impact during development, making me feel somewhat uneasy the whole time I was writing the game. It was, again, the right thing to do, but I should have used Starling more deeply before (which is the conundrum: I used it for Ludum Dare just so I could learn more about it). Argument and user experience One final aspect of the game that I learned is that making the game obvious for your players goes a long way into making it fun. If you have to spend the longest time explaining things, your game is doing something wrong. And that’s exactly the problem Survival of the Tastiest ultimately faced. It’s very hard for people to understand what’s going on with the game, why, and how. I did have some introductory text at the beginning, but that was a last-minute thing. More importantly, I should have had a better interface or simplified the whole concept so it would be easier for people to understand. That doesn’t mean the game itself should be simple. It just means that the experience and interface should be approachable and understandable. Conclusion I’m extremely happy with what I’ve done and, especially given that this was my first Ludum Dare. However, I feel like I’ve learned a lot of what not to do. The biggest problem is overscoping. Like Eric Decker said, the biggest lesson we can learn with this is probably with scoping – deciding what to do beforehand in a way you can complete it without having to rush and do something half-assed. I’m sure I will do more Ludum Dares in the future. But if there are any lessons I can take of it, they are to make it simple, to use frameworks and platforms you already have some absolute experience with (otherwise you’ll spend too much time trying to solve easy questions), and to scope for a game that you can complete in one day only (that way, you can actually take two days and make it cool). This entry was posted on Monday, August 27th, 2012 at 10:54 am and is filed under LD #24. You can follow any responses to this entry through the RSS 2.0 feed. You can skip to the end and leave a response. Pinging is currently not allowed. 3 Responses to ““Survival of the Tastiest” Post-mortem” darn it , knowing that I missed your livestream makes me a sad panda ;( but more to the point, the game is … well for a startup its original to say the least ;D it has some really neat ideas and more importantly its designed arround touch screens whitch by the looks of the submission is something rare ;o or that could be just me and my short memory -_-! awesum game, love et <3. Bolognese, for example, is very strong, but inaccurate; Pesto is very accurate and has great range, but it’s weaker; and my favorite, Vodka, would triggers 10% loss of speed on the pasta hit by it. The code for that is mostly in there. But in the end, I didn’t have time to implement the sauce selection interface; all pasta ended up using bolognese sauce. To-do list: lots of things were not done Actual programming also took a toll in the development time. Having been programming for a while, I like to believe I got to a point where I know how to make things right, but at the expense of forgetting how to do things wrong in a seemingly good way. What I mean is that I had to take a lot of shortcuts in my code to save time (e.g. a lot of singletons references for cross-communication rather than events or observers, all-encompassing check loops, not fast enough) that left a very sour taste in my mouth. While I know I used to do those a few years ago and survive, I almost cannot accept the state my code is in right now. At the same time, I do know it was the right thing to do given the timeframe. One small thing that had some impact was using a somewhat new platform for me. That’s Starling, the accelerated graphics framework I used in Flash. I had tested it before and I knew how to use it well – the API is very similar to Flash itself. However, there were some small details that had some impact during development, making me feel somewhat uneasy the whole time I was writing the game. It was, again, the right thing to do, but I should have used Starling more deeply before (which is the conundrum: I used it for Ludum Dare just so I could learn more about it). Argument and user experience One final aspect of the game that I learned is that making the game obvious for your players goes a long way into making it fun. If you have to spend the longest time explaining things, your game is doing something wrong. And that’s exactly the problem Survival of the Tastiest ultimately faced. It’s very hard for people to understand what’s going on with the game, why, and how. I did have some introductory text at the beginning, but that was a last-minute thing. More importantly, I should have had a better interface or simplified the whole concept so it would be easier for people to understand. That doesn’t mean the game itself should be simple. It just means that the experience and interface should be approachable and understandable. Conclusion I’m extremely happy with what I’ve done and, especially given that this was my first Ludum Dare. However, I feel like I’ve learned a lot of what not to do. The biggest problem is overscoping. Like Eric Decker said, the biggest lesson we can learn with this is probably with scoping – deciding what to do beforehand in a way you can complete it without having to rush and do something half-assed. I’m sure I will do more Ludum Dares in the future. But if there are any lessons I can take of it, they are to make it simple, to use frameworks and platforms you already have some absolute experience with (otherwise you’ll spend too much time trying to solve easy questions), and to scope for a game that you can complete in one day only (that way, you can actually take two days and make it cool). This entry was posted on Monday, August 27th, 2012 at 10:54 am and is filed under LD #24. You can follow any responses to this entry through the RSS 2.0 feed. You can skip to the end and leave a response. Pinging is currently not allowed. 3 Responses to ““Survival of the Tastiest” Post-mortem” darn it , knowing that I missed your livestream makes me a sad panda ;( but more to the point, the game is … well for a startup its original to say the least ;D it has some really neat ideas and more importantly its designed arround touch screens whitch by the looks of the submission is something rare ;o or that could be just me and my short memory -_-! awesum game, love et <3.  I did have some introductory text at the beginning, but that was a last-minute thing. More importantly, I should have had a better interface or simplified the whole concept so it would be easier for people to understand. I had tested it before and I knew how to use it well – the API is very similar to Flash itself. However, there were some small details that had some impact during development, making me feel somewhat uneasy the whole time I was writing the game."
    # ]
    # inputs = ["How big is the universe?"]
    resp_sentences, resp_sentences_seg = gpt_generate(inputs, 1)
    print(resp_sentences)
