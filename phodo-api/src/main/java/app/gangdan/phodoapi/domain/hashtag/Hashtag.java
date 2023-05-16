package app.gangdan.phodoapi.domain.hashtag;

import app.gangdan.phodoapi.domain.BaseEntity;
import app.gangdan.phodoapi.domain.member.Member;
import app.gangdan.phodoapi.domain.photoGuide.PhotoGuide;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import javax.persistence.*;

@Entity
@Table(name = "hashtag")
@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Hashtag extends BaseEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long hashtagId;

    private String hashtagName;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "photo_guide_id")
    private PhotoGuide photoGuide;

}
