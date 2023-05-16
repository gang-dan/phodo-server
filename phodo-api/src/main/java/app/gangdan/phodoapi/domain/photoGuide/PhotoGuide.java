package app.gangdan.phodoapi.domain.photoGuide;

import app.gangdan.phodoapi.domain.BaseEntity;
import app.gangdan.phodoapi.domain.photoSpot.PhotoSpot;
import lombok.*;

import javax.persistence.*;

@Entity
@Table(name = "photo_guide")
@Getter
@Builder
@NoArgsConstructor @AllArgsConstructor
public class PhotoGuide extends BaseEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long photoGuideId;

    private String PhotoGuideName;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "photo_spot_id")
    private PhotoSpot photoSpot;

}
