export const FEATURE_FIELDS = [
  {
    key: "follower_count",
    label: "Followers",
    min: 0,
    max: 50000,
    step: 1,
    group: "graph",
  },
  {
    key: "following_count",
    label: "Following",
    min: 0,
    max: 8000,
    step: 1,
    group: "graph",
  },
  {
    key: "friends_count",
    label: "Friends",
    min: 0,
    max: 5000,
    step: 1,
    group: "graph",
  },
  {
    key: "friend_request_mutuals",
    label: "Mutual connections",
    min: 0,
    max: 180,
    step: 1,
    group: "graph",
  },
  {
    key: "posts_count",
    label: "Posts",
    min: 0,
    max: 4000,
    step: 1,
    group: "activity",
  },
  {
    key: "account_age_days",
    label: "Account age (days)",
    min: 1,
    max: 4000,
    step: 1,
    group: "activity",
  },
  {
    key: "avg_likes_per_post",
    label: "Avg. likes per post",
    min: 0,
    max: 4000,
    step: 1,
    group: "activity",
  },
  {
    key: "posts_per_week",
    label: "Posts per week",
    min: 0,
    max: 30,
    step: 0.1,
    group: "activity",
  },
  {
    key: "bio_length",
    label: "Bio length",
    min: 0,
    max: 280,
    step: 1,
    group: "identity",
  },
];

export const DEFAULT_PROFILE = {
  username: "maya.nair",
  platform: "facebook",
  follower_count: 842,
  following_count: 361,
  friends_count: 478,
  posts_count: 216,
  account_age_days: 1480,
  has_profile_photo: 1,
  has_cover_photo: 1,
  bio_length: 96,
  username_digit_ratio: 0,
  username_length: 9,
  friend_request_mutuals: 18,
  profile_completeness: 0.92,
  avg_likes_per_post: 34,
  posts_per_week: 1.1,
};

export function digitRatio(username) {
  if (!username) return 0;
  const digits = [...username].filter((ch) => ch >= "0" && ch <= "9").length;
  return digits / username.length;
}

export function completeness(profile) {
  const parts = [
    Number(profile.has_profile_photo),
    Number(profile.has_cover_photo),
    Math.min(Math.max(profile.bio_length, 0) / 80, 1),
    Math.min(Math.max(profile.posts_count, 0) / 40, 1),
    profile.username_length >= 4 && profile.username_length <= 18 ? 1 : 0.35,
    Math.min(Math.max(profile.account_age_days, 0) / 365, 1),
  ];
  return parts.reduce((sum, value) => sum + value, 0) / parts.length;
}

export function deriveIdentity(profile) {
  const username = profile.username || "";
  const next = {
    ...profile,
    username_length: username.length,
    username_digit_ratio: Number(digitRatio(username).toFixed(4)),
  };
  next.profile_completeness = Number(completeness(next).toFixed(4));
  return next;
}

export function formatStat(value, key) {
  if (key === "username_digit_ratio" || key === "profile_completeness") {
    return `${Math.round(Number(value) * 100)}%`;
  }
  if (key === "posts_per_week" || key === "avg_likes_per_post") {
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

export function platformLabel(platform) {
  const map = { facebook: "Facebook", instagram: "Instagram", twitter: "Twitter" };
  return map[platform] || platform;
}
