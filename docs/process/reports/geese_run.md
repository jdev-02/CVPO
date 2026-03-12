# CVPO Run Report

- Generated: `2026-03-11T00:08:34.808690+00:00`
- Output Format: `pretty`

## Summary

```text
CVPO Guided Workflow
====================
Goal: geese_tracking
Experience: beginner
Frontend: cli - Fast, scriptable, and best for automation or power users.

Honest Assessment
-----------------
Summary: Moderate confidence for park geese concentration analysis.
Works Well:
  - Detects bird-like objects in open scenes.
  - Tracks object movement trends over time.
  - Produces time-series and concentration indicators.
Limitations:
  - Species-level accuracy may vary without specialized bird models.
  - ID switches can occur when birds cross paths.
  - Dense flocks may lead to undercounting.
Fit For Purpose:
  - Park pattern analysis
  - Operational trend monitoring
Not Fit For:
  - Scientific census-grade measurement
  - Regulatory reporting requiring species-certified precision
Citations: Lin et al. 2014 (COCO dataset), Zhang et al. 2022 (ByteTrack)

Socratic Check
--------------
Question: Why might tracking every frame be unnecessary for slow-moving geese, and what is the tradeoff when skipping frames?
Skipped: False
Answer: Slow motion allows frame subsampling while preserving trajectory continuity. Skipping frames reduces compute cost but increases risk of missed short events or ID switches during quick crossings.
Citations: Bewley et al. 2016 (SORT), Zhang et al. 2022 (ByteTrack)

Run Result
----------
{
  "workflow": "level3_detect_segment_classify_track",
  "frame_count": 8,
  "unique_track_ids": [
    1
  ],
  "input_video": null,
  "per_frame": [
    {
      "frame_index": 0,
      "detections": 1,
      "classes": [
        "goose"
      ],
      "track_ids": [
        1
      ]
    },
    {
      "frame_index": 1,
      "detections": 1,
      "classes": [
        "goose"
      ],
      "track_ids": [
        1
      ]
    },
    {
      "frame_index": 2,
      "detections": 1,
      "classes": [
        "goose"
      ],
      "track_ids": [
        1
      ]
    },
    {
      "frame_index": 3,
      "detections": 1,
      "classes": [
        "goose"
      ],
      "track_ids": [
        1
      ]
    },
    {
      "frame_index": 4,
      "detections": 1,
      "classes": [
        "goose"
      ],
      "track_ids": [
        1
      ]
    },
    {
      "frame_index": 5,
      "detections": 1,
      "classes": [
        "goose"
      ],
      "track_ids": [
        1
      ]
    },
    {
      "frame_index": 6,
      "detections": 1,
      "classes": [
        "goose"
      ],
      "track_ids": [
        1
      ]
    },
    {
      "frame_index": 7,
      "detections": 1,
      "classes": [
        "goose"
      ],
      "track_ids": [
        1
      ]
    }
  ]
}
```

## Raw JSON

```json
{
  "mode": "guided_workflow",
  "goal": "geese_tracking",
  "experience_level": "beginner",
  "frontend_choice": "cli",
  "frontend_description": "Fast, scriptable, and best for automation or power users.",
  "honest_assessment": {
    "summary": "Moderate confidence for park geese concentration analysis.",
    "works_well": [
      "Detects bird-like objects in open scenes.",
      "Tracks object movement trends over time.",
      "Produces time-series and concentration indicators."
    ],
    "limitations": [
      "Species-level accuracy may vary without specialized bird models.",
      "ID switches can occur when birds cross paths.",
      "Dense flocks may lead to undercounting."
    ],
    "fit_for_purpose": [
      "Park pattern analysis",
      "Operational trend monitoring"
    ],
    "not_fit_for": [
      "Scientific census-grade measurement",
      "Regulatory reporting requiring species-certified precision"
    ],
    "citations": [
      "Lin et al. 2014 (COCO dataset)",
      "Zhang et al. 2022 (ByteTrack)"
    ]
  },
  "socratic": {
    "question": "Why might tracking every frame be unnecessary for slow-moving geese, and what is the tradeoff when skipping frames?",
    "skipped": false,
    "answer": "Slow motion allows frame subsampling while preserving trajectory continuity. Skipping frames reduces compute cost but increases risk of missed short events or ID switches during quick crossings.",
    "citations": [
      "Bewley et al. 2016 (SORT)",
      "Zhang et al. 2022 (ByteTrack)"
    ]
  },
  "run_result": {
    "workflow": "level3_detect_segment_classify_track",
    "frame_count": 8,
    "unique_track_ids": [
      1
    ],
    "input_video": null,
    "per_frame": [
      {
        "frame_index": 0,
        "detections": 1,
        "classes": [
          "goose"
        ],
        "track_ids": [
          1
        ]
      },
      {
        "frame_index": 1,
        "detections": 1,
        "classes": [
          "goose"
        ],
        "track_ids": [
          1
        ]
      },
      {
        "frame_index": 2,
        "detections": 1,
        "classes": [
          "goose"
        ],
        "track_ids": [
          1
        ]
      },
      {
        "frame_index": 3,
        "detections": 1,
        "classes": [
          "goose"
        ],
        "track_ids": [
          1
        ]
      },
      {
        "frame_index": 4,
        "detections": 1,
        "classes": [
          "goose"
        ],
        "track_ids": [
          1
        ]
      },
      {
        "frame_index": 5,
        "detections": 1,
        "classes": [
          "goose"
        ],
        "track_ids": [
          1
        ]
      },
      {
        "frame_index": 6,
        "detections": 1,
        "classes": [
          "goose"
        ],
        "track_ids": [
          1
        ]
      },
      {
        "frame_index": 7,
        "detections": 1,
        "classes": [
          "goose"
        ],
        "track_ids": [
          1
        ]
      }
    ]
  }
}
```
