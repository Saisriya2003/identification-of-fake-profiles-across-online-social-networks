"""Synthetic multi-platform public-profile signals. No scraped identities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.features import FEATURE_NAMES, compute_profile_completeness, username_digit_ratio

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = BACKEND_ROOT / "data" / "profiles.csv"

FIRST = [
    "maya",
    "arjun",
    "elena",
    "kai",
    "priya",
    "noah",
    "amara",
    "leo",
    "sienna",
    "rohan",
    "isla",
    "dev",
    "nina",
    "omar",
    "lucia",
    "theo",
    "anika",
    "felix",
    "zara",
    "hugo",
    "meera",
    "jonas",
    "aisha",
    "marco",
    "leila",
    "samir",
    "freya",
    "aditya",
    "clara",
    "ravi",
]

LAST = [
    "rao",
    "chen",
    "okoye",
    "berg",
    "iyer",
    "santos",
    "nair",
    "vale",
    "d'Souza",
    "khan",
    "moreau",
    "park",
    "alvarez",
    "singh",
    "weiss",
    "costa",
    "nguyen",
    "kade",
    "hassan",
    "brook",
]

FAKE_STEMS = [
    "user",
    "official",
    "real",
    "promo",
    "acc",
    "follow",
    "win",
    "deal",
    "net",
    "id",
    "tmp",
    "new",
    "bot",
    "xx",
    "q",
]


def _lognormal_clip(rng: np.random.Generator, n: int, mean: float, sigma: float, lo: float, hi: float) -> np.ndarray:
    values = rng.lognormal(mean, sigma, size=n)
    return np.clip(values, lo, hi)


def _make_usernames(rng: np.random.Generator, is_fake: np.ndarray) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for fake in is_fake:
        if fake:
            if rng.random() < 0.18:
                first = str(rng.choice(FIRST))
                last = str(rng.choice(LAST)).replace("'", "").replace(" ", "")
                candidate = f"{first}{last}{int(rng.integers(1, 9))}"[:24]
            else:
                stem = str(rng.choice(FAKE_STEMS))
                digits = "".join(str(int(rng.integers(0, 10))) for _ in range(int(rng.integers(3, 8))))
                sep = str(rng.choice(["", "_", "."]))
                extra = str(rng.choice(["", str(rng.choice(FAKE_STEMS))]))
                candidate = f"{stem}{sep}{extra}{digits}"[:28]
        else:
            first = str(rng.choice(FIRST))
            last = str(rng.choice(LAST)).replace("'", "").replace(" ", "")
            sep = str(rng.choice(["", ".", "_"]))
            year = str(int(rng.integers(90, 100))) if rng.random() < 0.12 else ""
            candidate = f"{first}{sep}{last}{year}"[:24]
        base = candidate.lower()
        name = base
        n = 2
        while name in seen:
            name = f"{base}{n}"
            n += 1
        seen.add(name)
        names.append(name)
    return names


def generate(n: int = 4000, seed: int = 2025) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    platforms = rng.choice(
        np.array(["facebook", "instagram", "twitter"]),
        size=n,
        p=[0.42, 0.36, 0.22],
    )
    is_fake = rng.random(n) < 0.44

    # Overlapping subpopulations so the task is not linearly trivial.
    sockpuppet = is_fake & (rng.random(n) < 0.14)
    new_genuine = (~is_fake) & (rng.random(n) < 0.16)
    quiet_real = (~is_fake) & (rng.random(n) < 0.10)
    groomed_fake = is_fake & (rng.random(n) < 0.12)

    age = np.empty(n, dtype=np.float64)
    age[is_fake] = _lognormal_clip(rng, int(is_fake.sum()), 3.15, 0.85, 1, 220)
    age[~is_fake] = _lognormal_clip(rng, int((~is_fake).sum()), 6.55, 0.72, 40, 5200)
    age[sockpuppet] = rng.uniform(380, 2100, size=int(sockpuppet.sum()))
    age[new_genuine] = rng.uniform(4, 55, size=int(new_genuine.sum()))
    age = np.rint(age).astype(np.int64)

    followers = np.empty(n, dtype=np.float64)
    following = np.empty(n, dtype=np.float64)
    friends = np.empty(n, dtype=np.float64)
    posts = np.empty(n, dtype=np.float64)
    likes = np.empty(n, dtype=np.float64)
    mutuals = np.empty(n, dtype=np.float64)
    bio = np.empty(n, dtype=np.float64)
    has_photo = np.zeros(n, dtype=np.int64)
    has_cover = np.zeros(n, dtype=np.int64)

    for platform in ("facebook", "instagram", "twitter"):
        mask = platforms == platform
        fake_m = mask & is_fake
        real_m = mask & ~is_fake
        nf, nr = int(fake_m.sum()), int(real_m.sum())

        if platform == "instagram":
            followers[real_m] = _lognormal_clip(rng, nr, 6.4, 1.15, 40, 80_000)
            followers[fake_m] = _lognormal_clip(rng, nf, 3.1, 0.9, 0, 420)
            following[real_m] = _lognormal_clip(rng, nr, 5.6, 0.55, 40, 4_500)
            following[fake_m] = _lognormal_clip(rng, nf, 7.2, 0.45, 400, 8_500)
            friends[real_m] = _lognormal_clip(rng, nr, 3.4, 0.7, 0, 800)
            friends[fake_m] = _lognormal_clip(rng, nf, 2.0, 0.8, 0, 80)
            posts[real_m] = _lognormal_clip(rng, nr, 4.6, 0.85, 8, 4_500)
            posts[fake_m] = _lognormal_clip(rng, nf, 1.4, 0.9, 0, 28)
            likes[real_m] = _lognormal_clip(rng, nr, 4.2, 1.0, 4, 6_000)
            likes[fake_m] = _lognormal_clip(rng, nf, 1.1, 0.8, 0, 18)
            photo_p, cover_p = 0.96, 0.55
        elif platform == "facebook":
            followers[real_m] = _lognormal_clip(rng, nr, 5.2, 0.9, 20, 25_000)
            followers[fake_m] = _lognormal_clip(rng, nf, 2.6, 0.85, 0, 180)
            following[real_m] = _lognormal_clip(rng, nr, 5.0, 0.5, 20, 2_800)
            following[fake_m] = _lognormal_clip(rng, nf, 6.6, 0.5, 220, 5_500)
            friends[real_m] = _lognormal_clip(rng, nr, 5.9, 0.55, 80, 4_800)
            friends[fake_m] = _lognormal_clip(rng, nf, 3.1, 0.7, 4, 140)
            posts[real_m] = _lognormal_clip(rng, nr, 4.3, 0.8, 6, 3_200)
            posts[fake_m] = _lognormal_clip(rng, nf, 1.2, 0.85, 0, 22)
            likes[real_m] = _lognormal_clip(rng, nr, 3.1, 0.85, 2, 1_200)
            likes[fake_m] = _lognormal_clip(rng, nf, 0.7, 0.7, 0, 12)
            photo_p, cover_p = 0.94, 0.82
        else:  # twitter
            followers[real_m] = _lognormal_clip(rng, nr, 5.5, 1.1, 15, 40_000)
            followers[fake_m] = _lognormal_clip(rng, nf, 2.4, 0.95, 0, 220)
            following[real_m] = _lognormal_clip(rng, nr, 5.4, 0.6, 25, 4_000)
            following[fake_m] = _lognormal_clip(rng, nf, 7.0, 0.5, 350, 9_000)
            friends[real_m] = _lognormal_clip(rng, nr, 2.8, 0.8, 0, 400)
            friends[fake_m] = _lognormal_clip(rng, nf, 1.4, 0.7, 0, 40)
            posts[real_m] = _lognormal_clip(rng, nr, 5.4, 0.95, 12, 8_000)
            posts[fake_m] = _lognormal_clip(rng, nf, 1.8, 1.0, 0, 40)
            likes[real_m] = _lognormal_clip(rng, nr, 2.6, 0.9, 0, 800)
            likes[fake_m] = _lognormal_clip(rng, nf, 0.5, 0.65, 0, 8)
            photo_p, cover_p = 0.88, 0.42

        has_photo[real_m] = (rng.random(nr) < photo_p).astype(np.int64)
        has_photo[fake_m] = (rng.random(nf) < 0.42).astype(np.int64)
        has_cover[real_m] = (rng.random(nr) < cover_p).astype(np.int64)
        has_cover[fake_m] = (rng.random(nf) < 0.18).astype(np.int64)

        bio[real_m] = np.clip(rng.normal(88, 48, size=nr), 0, 280)
        bio[fake_m] = np.clip(rng.normal(22, 20, size=nf), 0, 90)

        mutuals[real_m] = _lognormal_clip(rng, nr, 2.7, 0.75, 1, 180)
        mutuals[fake_m] = _lognormal_clip(rng, nf, 0.35, 0.7, 0, 6)

    # New genuine accounts look thin; sockpuppets look slightly more complete.
    posts[new_genuine] = np.clip(posts[new_genuine] * 0.15, 0, 18)
    likes[new_genuine] = np.clip(likes[new_genuine] * 0.25, 0, 40)
    mutuals[new_genuine] = np.clip(mutuals[new_genuine] * 0.2, 0, 8)
    posts[sockpuppet] = np.clip(posts[sockpuppet] + rng.uniform(8, 40, size=int(sockpuppet.sum())), 0, 120)
    has_photo[sockpuppet] = 1

    posts[quiet_real] = np.clip(posts[quiet_real] * 0.08, 0, 12)
    likes[quiet_real] = np.clip(likes[quiet_real] * 0.12, 0, 20)
    bio[quiet_real] = np.clip(bio[quiet_real] * 0.15, 0, 24)

    n_groomed = int(groomed_fake.sum())
    if n_groomed:
        followers[groomed_fake] = np.clip(
            followers[groomed_fake] * 6 + rng.uniform(60, 380, size=n_groomed), 20, 2500
        )
        posts[groomed_fake] = rng.uniform(18, 90, size=n_groomed)
        likes[groomed_fake] = rng.uniform(4, 40, size=n_groomed)
        bio[groomed_fake] = rng.uniform(36, 150, size=n_groomed)
        mutuals[groomed_fake] = rng.uniform(1, 16, size=n_groomed)
        has_photo[groomed_fake] = 1
        has_cover[groomed_fake] = (rng.random(n_groomed) < 0.55).astype(np.int64)

    for arr in (followers, following, friends, posts, likes, mutuals):
        arr *= np.clip(rng.normal(1.0, 0.14, size=n), 0.55, 1.6)

    usernames = _make_usernames(rng, is_fake)
    digit_ratio = np.array([username_digit_ratio(u) for u in usernames], dtype=np.float64)
    uname_len = np.array([len(u) for u in usernames], dtype=np.float64)

    weeks = np.maximum(age / 7.0, 0.35)
    posts_per_week = posts / weeks
    # Cold-start fakes sometimes burst a handful of posts in week one.
    burst = is_fake & (age < 21) & (rng.random(n) < 0.45)
    posts_per_week[burst] = rng.uniform(4.0, 18.0, size=int(burst.sum()))
    posts_per_week = np.clip(posts_per_week, 0, 40)

    completeness = np.array(
        [
            compute_profile_completeness(
                float(has_photo[i]),
                float(has_cover[i]),
                float(bio[i]),
                float(posts[i]),
                float(uname_len[i]),
                float(age[i]),
            )
            for i in range(n)
        ],
        dtype=np.float64,
    )

    frame = pd.DataFrame(
        {
            "username": usernames,
            "platform": platforms,
            "label": np.where(is_fake, "fake", "real"),
            "follower_count": np.rint(followers).astype(int),
            "following_count": np.rint(following).astype(int),
            "friends_count": np.rint(friends).astype(int),
            "posts_count": np.rint(posts).astype(int),
            "account_age_days": age.astype(int),
            "has_profile_photo": has_photo.astype(int),
            "has_cover_photo": has_cover.astype(int),
            "bio_length": np.rint(bio).astype(int),
            "username_digit_ratio": np.round(digit_ratio, 4),
            "username_length": uname_len.astype(int),
            "friend_request_mutuals": np.rint(mutuals).astype(int),
            "profile_completeness": np.round(completeness, 4),
            "avg_likes_per_post": np.round(likes, 2),
            "posts_per_week": np.round(posts_per_week, 3),
        }
    )
    missing = [c for c in FEATURE_NAMES if c not in frame.columns]
    if missing:
        raise RuntimeError(f"generated frame missing features: {missing}")
    return frame


def generate_and_save(n: int = 4000, seed: int = 2025, path: Path | None = None) -> Path:
    dest = path or DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame = generate(n=n, seed=seed)
    frame.to_csv(dest, index=False)
    print(f"wrote {len(frame)} profiles -> {dest}")
    print(frame["label"].value_counts().to_string())
    print(frame["platform"].value_counts().to_string())
    return dest


if __name__ == "__main__":
    generate_and_save()
